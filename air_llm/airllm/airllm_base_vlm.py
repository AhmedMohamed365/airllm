
from .airllm_base import AirLLMBaseModel

from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers import AutoProcessor
from transformers.quantizers import AutoHfQuantizer

from .utils import clean_memory


class AirLLMBaseVLM(AirLLMBaseModel):
    """
    Base class for Vision Language Models in AirLLM.

    Extends AirLLMBaseModel to handle image inputs alongside text.
    The vision encoder and projector are loaded to GPU, process images,
    then freed. The LLM backbone uses layer-by-layer processing.
    """

    def __init__(self, *args, **kwargs):
        super(AirLLMBaseVLM, self).__init__(*args, **kwargs)

        # Prepend VLM layer names to the layer_names list
        vlm_names = [self.layer_names_dict[k] for k in self.get_vlm_extra_layer_keys()]
        self.layer_names = vlm_names + self.layer_names

        # Get processor (handles both text and images)
        self.processor = self.get_processor(hf_token=self.hf_token)

        # Temporary storage for vision features during forward pass
        self._image_features = None
        self._pixel_values = None

    def get_vlm_extra_layer_keys(self):
        """Return keys in layer_names_dict for VLM-specific components."""
        raise NotImplementedError

    def get_auto_model_class(self):
        """Return the model class for this VLM (e.g., LlavaForConditionalGeneration)."""
        raise NotImplementedError

    def get_processor(self, hf_token=None):
        if hf_token is not None:
            return AutoProcessor.from_pretrained(
                self.model_local_path, token=hf_token, trust_remote_code=True)
        return AutoProcessor.from_pretrained(
            self.model_local_path, trust_remote_code=True)

    def get_use_better_transformer(self):
        return False

    def init_model(self):
        """Override to create VLM model instead of CausalLM."""
        model_class = self.get_auto_model_class()

        self.model = None
        with init_empty_weights():
            self.model = model_class.from_config(self.config, trust_remote_code=True)

        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(
                quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(
                model=self.model, device_map=device_map)

        self.model.eval()
        self.model.tie_weights()

        self.set_layers_from_layer_names()

        # Move buffers to device
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model, buffer_name, self.running_device,
                value=buffer, dtype=self.running_dtype)

    def set_layers_from_layer_names(self):
        """Override to include VLM-specific layers."""
        super().set_layers_from_layer_names()

        # Prepend VLM layers (vision tower, projector, etc.)
        vlm_layers = []
        for key in self.get_vlm_extra_layer_keys():
            name = self.layer_names_dict[key]
            model_attr = self.model
            for attr_name in name.split("."):
                model_attr = getattr(model_attr, attr_name)
            vlm_layers.append(model_attr)
        self.layers = vlm_layers + self.layers

    def is_vlm_layer(self, layer_name):
        vlm_layer_names = [self.layer_names_dict[k]
                           for k in self.get_vlm_extra_layer_keys()]
        return layer_name in vlm_layer_names

    def process_vlm_layer(self, layer_name, layer):
        """Process VLM-specific layers using stored pixel_values."""
        raise NotImplementedError

    def run_embed(self, layer, seq):
        """Override to merge vision features with text embeddings."""
        text_embeds = layer(seq)
        if self._image_features is not None:
            text_embeds = self._merge_input_ids_with_image_features(
                self._image_features, text_embeds, seq)
        return text_embeds

    def _merge_input_ids_with_image_features(
            self, image_features, inputs_embeds, input_ids):
        """Merge image features into text embeddings at image token positions."""
        raise NotImplementedError

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None,
            attention_mask=None, inputs_embeds=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            attention_mask=attention_mask, inputs_embeds=inputs_embeds,
            **kwargs)
        if 'pixel_values' in kwargs:
            model_inputs['pixel_values'] = kwargs['pixel_values']
        if 'image_grid_thw' in kwargs:
            model_inputs['image_grid_thw'] = kwargs['image_grid_thw']
        return model_inputs

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        # Store pixel_values for VLM layer processing
        self._pixel_values = pixel_values
        self._image_features = None

        result = super().forward(input_ids=input_ids, **kwargs)

        # Cleanup
        self._pixel_values = None
        self._image_features = None

        return result
