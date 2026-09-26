"""Shared KoELECTRA + BiLSTM + CRF model definition."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import ElectraModel

try:
    from torchcrf import CRF
except ImportError as exc:  # pragma: no cover - import guard is exercised by runtime envs.
    CRF = None
    _CRF_IMPORT_ERROR = exc
else:
    _CRF_IMPORT_ERROR = None


class KoElectraNER(nn.Module):
    """Checkpoint-compatible model used by the published scripts.

    Attribute names intentionally stay ``electra``, ``lstm``, ``dropout``,
    ``classifier``, and ``crf`` so existing raw state dicts load without key
    rewriting.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        *,
        lstm_hidden_size: int = 256,
        dropout: float = 0.1,
        o_label_id: int = 12,
        num_samples: int | None = None,
    ) -> None:
        super().__init__()
        if CRF is None:
            raise ImportError("torchcrf is required to instantiate KoElectraNER") from _CRF_IMPORT_ERROR
        self.num_labels = num_labels
        self.num_samples = num_samples
        self.o_label_id = o_label_id
        self.electra = ElectraModel.from_pretrained(model_name)
        hidden_size = self.electra.config.hidden_size
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        return self.classifier(self.dropout(lstm_output))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        output_attentions: bool = False,
    ) -> Any:
        electra_kwargs = {"output_attentions": output_attentions} if output_attentions else {}
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, **electra_kwargs)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        emissions = self.classifier(self.dropout(lstm_output))
        crf_emissions = emissions.float()
        mask = attention_mask.bool()

        loss = None
        if labels is not None:
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = self.o_label_id
            loss = -self.crf(crf_emissions, labels_crf, mask=mask, reduction="mean")
            return loss, emissions

        predictions = self.crf.decode(crf_emissions, mask=mask)
        if output_attentions:
            result = (predictions, outputs.attentions)
        else:
            result = predictions
        return result
