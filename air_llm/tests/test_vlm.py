import unittest
from unittest.mock import patch, MagicMock

from airllm.auto_model import AutoModel


class TestAutoModelVLM(unittest.TestCase):
    """Test VLM architecture detection in AutoModel."""

    def _mock_get_module_class(self, architecture_name):
        """Helper to test architecture detection with mocked config."""
        mock_config = MagicMock()
        mock_config.architectures = [architecture_name]

        with patch('airllm.auto_model.AutoConfig') as MockAutoConfig:
            MockAutoConfig.from_pretrained.return_value = mock_config
            return AutoModel.get_module_class('dummy/model')

    def test_llava_detection(self):
        module, cls = self._mock_get_module_class(
            'LlavaForConditionalGeneration')
        self.assertEqual(cls, 'AirLLMLlava')

    def test_llava_next_detection(self):
        module, cls = self._mock_get_module_class(
            'LlavaNextForConditionalGeneration')
        self.assertEqual(cls, 'AirLLMLlava')

    def test_qwen2_vl_detection(self):
        module, cls = self._mock_get_module_class(
            'Qwen2VLForConditionalGeneration')
        self.assertEqual(cls, 'AirLLMQwen2VL')

    def test_existing_text_models_still_work(self):
        """Ensure VLM additions don't break existing text model detection."""
        text_models = {
            'Qwen2ForCausalLM': 'AirLLMQWen2',
            'QWenLMHeadModel': 'AirLLMQWen',
            'BaichuanForCausalLM': 'AirLLMBaichuan',
            'ChatGLMForConditionalGeneration': 'AirLLMChatGLM',
            'InternLMForCausalLM': 'AirLLMInternLM',
            'MistralForCausalLM': 'AirLLMMistral',
            'MixtralForCausalLM': 'AirLLMMixtral',
            'LlamaForCausalLM': 'AirLLMLlama2',
        }
        for arch, expected_cls in text_models.items():
            module, cls = self._mock_get_module_class(arch)
            self.assertEqual(cls, expected_cls,
                             f"Architecture {arch} should map to {expected_cls}")

    def test_vlm_base_class_inheritance(self):
        """Test that VLM classes properly inherit from the VLM base."""
        from airllm.airllm_base_vlm import AirLLMBaseVLM
        from airllm.airllm_llava import AirLLMLlava
        from airllm.airllm_qwen2_vl import AirLLMQwen2VL
        from airllm.airllm_base import AirLLMBaseModel

        self.assertTrue(issubclass(AirLLMBaseVLM, AirLLMBaseModel))
        self.assertTrue(issubclass(AirLLMLlava, AirLLMBaseVLM))
        self.assertTrue(issubclass(AirLLMQwen2VL, AirLLMBaseVLM))

    def test_vlm_base_is_vlm_layer_default(self):
        """Test that base model returns False for is_vlm_layer."""
        from airllm.airllm_base import AirLLMBaseModel
        base = AirLLMBaseModel.__new__(AirLLMBaseModel)
        self.assertFalse(base.is_vlm_layer('any_layer'))

    def test_llava_layer_names(self):
        """Test LLaVA layer name configuration."""
        from airllm.airllm_llava import AirLLMLlava
        llava = AirLLMLlava.__new__(AirLLMLlava)
        llava.set_layer_names_dict()

        self.assertEqual(
            llava.layer_names_dict['embed'],
            'language_model.model.embed_tokens')
        self.assertEqual(
            llava.layer_names_dict['layer_prefix'],
            'language_model.model.layers')
        self.assertEqual(
            llava.layer_names_dict['vision_model'],
            'vision_tower')
        self.assertEqual(
            llava.layer_names_dict['multi_modal_projector'],
            'multi_modal_projector')

    def test_qwen2_vl_layer_names(self):
        """Test Qwen2-VL layer name configuration."""
        from airllm.airllm_qwen2_vl import AirLLMQwen2VL
        qwen = AirLLMQwen2VL.__new__(AirLLMQwen2VL)
        qwen.set_layer_names_dict()

        self.assertEqual(
            qwen.layer_names_dict['embed'],
            'model.embed_tokens')
        self.assertEqual(
            qwen.layer_names_dict['visual'],
            'visual')

    def test_llava_vlm_extra_layer_keys(self):
        """Test LLaVA returns correct VLM extra layer keys."""
        from airllm.airllm_llava import AirLLMLlava
        llava = AirLLMLlava.__new__(AirLLMLlava)
        keys = llava.get_vlm_extra_layer_keys()
        self.assertEqual(keys, ['vision_model', 'multi_modal_projector'])

    def test_qwen2_vl_extra_layer_keys(self):
        """Test Qwen2-VL returns correct VLM extra layer keys."""
        from airllm.airllm_qwen2_vl import AirLLMQwen2VL
        qwen = AirLLMQwen2VL.__new__(AirLLMQwen2VL)
        keys = qwen.get_vlm_extra_layer_keys()
        self.assertEqual(keys, ['visual'])

    def test_llava_merge_embeddings(self):
        """Test LLaVA image-text embedding merge logic."""
        import torch
        from airllm.airllm_llava import AirLLMLlava

        llava = AirLLMLlava.__new__(AirLLMLlava)
        llava.config = MagicMock()
        llava.config.image_token_index = 32000

        # Simulate: 1 batch, 5 tokens, embed_dim=4
        # Tokens: [text, text, IMAGE, text, text]
        input_ids = torch.tensor([[1, 2, 32000, 3, 4]])
        inputs_embeds = torch.ones(1, 5, 4) * 0.5  # text embeds all 0.5

        # 1 image with 3 patches
        image_features = torch.ones(1, 3, 4) * 1.0  # image features all 1.0

        merged = llava._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids)

        # Result should be: 2 text + 3 image + 2 text = 7 tokens
        self.assertEqual(merged.shape, (1, 7, 4))
        # First 2 tokens should be text (0.5)
        self.assertTrue(torch.all(merged[0, :2] == 0.5))
        # Middle 3 tokens should be image (1.0)
        self.assertTrue(torch.all(merged[0, 2:5] == 1.0))
        # Last 2 tokens should be text (0.5)
        self.assertTrue(torch.all(merged[0, 5:] == 0.5))

    def test_llava_merge_no_image_tokens(self):
        """Test merge when there are no image tokens."""
        import torch
        from airllm.airllm_llava import AirLLMLlava

        llava = AirLLMLlava.__new__(AirLLMLlava)
        llava.config = MagicMock()
        llava.config.image_token_index = 32000

        input_ids = torch.tensor([[1, 2, 3, 4]])
        inputs_embeds = torch.ones(1, 4, 4) * 0.5
        image_features = torch.ones(1, 3, 4) * 1.0

        merged = llava._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids)

        # No image tokens, output should match input
        self.assertEqual(merged.shape, (1, 4, 4))
        self.assertTrue(torch.all(merged == 0.5))

    def test_qwen2_vl_merge_embeddings(self):
        """Test Qwen2-VL image-text embedding merge logic."""
        import torch
        from airllm.airllm_qwen2_vl import AirLLMQwen2VL

        qwen = AirLLMQwen2VL.__new__(AirLLMQwen2VL)
        qwen.config = MagicMock()
        qwen.config.image_token_id = 151655
        qwen.config.video_token_id = 151656

        # Simulate: 1 batch, 5 tokens, embed_dim=4
        # Tokens: [text, IMAGE_PAD, IMAGE_PAD, text, text]
        input_ids = torch.tensor([[1, 151655, 151655, 3, 4]])
        inputs_embeds = torch.ones(1, 5, 4) * 0.5

        # 2 image tokens worth of features
        image_features = torch.ones(2, 4) * 1.0

        merged = qwen._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids)

        # Same shape (Qwen2-VL replaces in-place, no expansion)
        self.assertEqual(merged.shape, (1, 5, 4))
        # Text tokens stay the same
        self.assertTrue(torch.all(merged[0, 0] == 0.5))
        self.assertTrue(torch.all(merged[0, 3:] == 0.5))
        # Image tokens replaced
        self.assertTrue(torch.all(merged[0, 1:3] == 1.0))


if __name__ == '__main__':
    unittest.main()
