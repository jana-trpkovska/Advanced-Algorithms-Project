import torch.nn as nn
from transformers import AutoModel


class ClinicalBERTClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int = 2):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name,
            use_safetensors=True
        )

        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits
