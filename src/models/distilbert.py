import torch.nn as nn
from transformers import AutoModel


class DistilBERTClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name="distilbert-base-uncased",
        num_labels=2,
        dropout_prob=0.3,
    ):
        super(DistilBERTClassifier, self).__init__()

        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0, :]

        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits
