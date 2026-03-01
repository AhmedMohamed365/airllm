
import torch
from transformers import LlavaForConditionalGeneration

from .airllm_base_vlm import AirLLMBaseVLM


class AirLLMLlava(AirLLMBaseVLM):
    """
    AirLLM implementation for LLaVA and LLaVA-NeXT vision-language models.

    Supports running large LLaVA models on low VRAM GPUs by loading
    vision encoder, projector, and LLM layers one at a time.
    """

    def __init__(self, *args, **kwargs):
        super(AirLLMLlava, self).__init__(*args, **kwargs)

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'language_model.model.embed_tokens',
            'layer_prefix': 'language_model.model.layers',
            'norm': 'language_model.model.norm',
            'lm_head': 'language_model.lm_head',
            'vision_model': 'vision_tower',
            'multi_modal_projector': 'multi_modal_projector',
        }

    def get_vlm_extra_layer_keys(self):
        return ['vision_model', 'multi_modal_projector']

    def get_auto_model_class(self):
        return LlavaForConditionalGeneration

    def process_vlm_layer(self, layer_name, layer):
        pixel_values = self._pixel_values
        if pixel_values is None:
            return

        if layer_name == self.layer_names_dict['vision_model']:
            outputs = layer(
                pixel_values.to(
                    device=self.device, dtype=self.running_dtype))
            self._image_features = outputs.last_hidden_state

            # Select features based on config
            select_strategy = getattr(
                self.config, 'vision_feature_select_strategy', 'default')
            if select_strategy == 'default':
                self._image_features = self._image_features[:, 1:]
            layer_idx = getattr(
                self.config, 'vision_feature_layer', None)
            if layer_idx is not None and hasattr(outputs, 'hidden_states'):
                if outputs.hidden_states is not None:
                    self._image_features = outputs.hidden_states[layer_idx]
                    if select_strategy == 'default':
                        self._image_features = self._image_features[:, 1:]

        elif layer_name == self.layer_names_dict['multi_modal_projector']:
            if self._image_features is not None:
                self._image_features = layer(
                    self._image_features.to(
                        device=self.device, dtype=self.running_dtype))

    def _merge_input_ids_with_image_features(
            self, image_features, inputs_embeds, input_ids):
        """LLaVA-style merge: replace image tokens with projected features."""
        image_token_index = getattr(self.config, 'image_token_index', 32000)
        batch_size, seq_len, embed_dim = inputs_embeds.shape

        new_embeds_list = []
        for b in range(batch_size):
            cur_ids = input_ids[b]
            cur_embeds = inputs_embeds[b]

            image_positions = (cur_ids == image_token_index
                               ).nonzero(as_tuple=True)[0]

            if len(image_positions) == 0:
                new_embeds_list.append(cur_embeds.unsqueeze(0))
                continue

            segments = []
            prev_pos = 0
            img_idx = 0
            for pos in image_positions:
                pos_val = pos.item()
                if pos_val > prev_pos:
                    segments.append(cur_embeds[prev_pos:pos_val])
                if img_idx < image_features.shape[0]:
                    segments.append(
                        image_features[img_idx].to(
                            device=cur_embeds.device,
                            dtype=cur_embeds.dtype))
                    img_idx += 1
                prev_pos = pos_val + 1

            if prev_pos < seq_len:
                segments.append(cur_embeds[prev_pos:])

            new_embeds = torch.cat(segments, dim=0)
            new_embeds_list.append(new_embeds.unsqueeze(0))

        # Pad to same length for batched processing
        max_len = max(e.shape[1] for e in new_embeds_list)
        padded = []
        for e in new_embeds_list:
            if e.shape[1] < max_len:
                pad = torch.zeros(
                    1, max_len - e.shape[1], embed_dim,
                    device=e.device, dtype=e.dtype)
                e = torch.cat([e, pad], dim=1)
            padded.append(e)

        return torch.cat(padded, dim=0)
