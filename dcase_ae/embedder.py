import numpy as np
import torch


class PretrainedAudioEmbedder:
    def __init__(self, model_name: str, sample_rate: int, device: torch.device):
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "The pretrained embedding backend requires transformers. "
                "Install it with: pip install transformers"
            ) from exc

        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def extract(self, waveforms: list[np.ndarray]) -> np.ndarray:
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        embeddings = self._mean_pool(outputs, inputs)
        return embeddings.detach().cpu().numpy().astype(np.float32, copy=False)

    def _mean_pool(self, outputs, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            raise RuntimeError(f"Model {self.model_name} did not return hidden states or pooled output.")

        hidden_mask = self._hidden_attention_mask(hidden, inputs)
        if hidden_mask is None:
            return hidden.mean(dim=1)
        masked_hidden = hidden * hidden_mask.unsqueeze(-1)
        denominator = hidden_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return masked_hidden.sum(dim=1) / denominator

    def _hidden_attention_mask(
        self,
        hidden: torch.Tensor,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            return None
        if hasattr(self.model, "_get_feat_extract_output_lengths"):
            input_lengths = attention_mask.sum(dim=1)
            output_lengths = self.model._get_feat_extract_output_lengths(input_lengths)
            output_lengths = output_lengths.to(device=hidden.device, dtype=torch.long)
        else:
            output_lengths = attention_mask.new_full(
                (attention_mask.size(0),),
                hidden.size(1),
                dtype=torch.long,
            )
        positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0)
        return positions < output_lengths.unsqueeze(1)
