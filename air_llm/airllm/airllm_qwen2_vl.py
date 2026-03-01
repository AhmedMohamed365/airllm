
import torch
from transformers import Qwen2VLForConditionalGeneration

from .airllm_base_vlm import AirLLMBaseVLM


class AirLLMQwen2VL(AirLLMBaseVLM):
    """
    AirLLM implementation for Qwen2-VL vision-language models.

    Supports running large Qwen2-VL models on low VRAM GPUs by loading
    vision encoder and LLM layers one at a time.
    """

    def __init__(self, *args, **kwargs):
        super(AirLLMQwen2VL, self).__init__(*args, **kwargs)

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
            'visual': 'visual',
        }

    def get_vlm_extra_layer_keys(self):
        return ['visual']

    def get_auto_model_class(self):
        return Qwen2VLForConditionalGeneration

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None,
            attention_mask=None, inputs_embeds=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            attention_mask=attention_mask, inputs_embeds=inputs_embeds,
            **kwargs)
        if 'image_grid_thw' in kwargs:
            model_inputs['image_grid_thw'] = kwargs['image_grid_thw']
        return model_inputs

    def forward(self, input_ids=None, pixel_values=None,
                image_grid_thw=None, **kwargs):
        self._image_grid_thw = image_grid_thw

        result = super().forward(
            input_ids=input_ids, pixel_values=pixel_values, **kwargs)

        self._image_grid_thw = None

        return result

    def process_vlm_layer(self, layer_name, layer):
        pixel_values = self._pixel_values
        if pixel_values is None:
            return

        if layer_name == self.layer_names_dict['visual']:
            grid_thw = getattr(self, '_image_grid_thw', None)
            self._image_features = layer(
                pixel_values.to(
                    device=self.device, dtype=self.running_dtype),
                grid_thw=grid_thw)

    def _merge_input_ids_with_image_features(
            self, image_features, inputs_embeds, input_ids):
        """Qwen2-VL merge: replace image pad tokens with visual features."""
        image_token_id = getattr(self.config, 'image_token_id', 151655)
        video_token_id = getattr(self.config, 'video_token_id', 151656)

        batch_size, seq_len, embed_dim = inputs_embeds.shape
        merged = inputs_embeds.clone()

        for b in range(batch_size):
            cur_ids = input_ids[b]
            mask = (cur_ids == image_token_id) | (cur_ids == video_token_id)
            n_vision_tokens = mask.sum().item()

            if n_vision_tokens == 0:
                continue

            vision_features = image_features.reshape(
                -1, image_features.shape[-1])

            n_available = min(n_vision_tokens, vision_features.shape[0])
            merged[b, mask] = vision_features[:n_available].to(
                device=merged.device, dtype=merged.dtype)

        return merged
