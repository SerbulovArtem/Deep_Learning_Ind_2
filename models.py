import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class GelectraClassifier(nn.Module):
    """GELECTRA-based classifier wired for Trainer.

    Expects `xb` to be a dict with keys like `input_ids`,
    `attention_mask`, etc. When `y` is provided, returns
    (logits, loss). When `y` is None, returns logits only.
    """

    def __init__(self, model_name: str = "deepset/gelectra-base", num_labels: int = 9):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3,
        )

    def forward(self, xb, y=None):
        if not isinstance(xb, dict):
            raise ValueError("GelectraClassifier expects xb to be a dict of tokenized inputs.")

        inputs = xb
        if y is not None:
            # Trainer passes y as a tensor of label indices
            outputs = self.backbone(**inputs, labels=y)
            logits = outputs.logits
            loss = outputs.loss
            return logits, loss

        # Inference path (e.g. submission): just return logits
        outputs = self.backbone(**inputs)
        return outputs.logits


class GottBERTClassifier(nn.Module):
    """GottBERT-based classifier wired for Trainer.

    Expects `xb` to be a dict with keys like `input_ids`,
    `attention_mask`, etc. When `y` is provided, returns
    (logits, loss). When `y` is None, returns logits only.
    """

    def __init__(self, model_name: str = "uklfr/gottbert-base", num_labels: int = 9):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3,
        )

    def forward(self, xb, y=None):
        if not isinstance(xb, dict):
            raise ValueError("GottBERTClassifier expects xb to be a dict of tokenized inputs.")

        inputs = xb
        if y is not None:
            # Trainer passes y as a tensor of label indices
            outputs = self.backbone(**inputs, labels=y)
            logits = outputs.logits
            loss = outputs.loss
            return logits, loss

        # Inference path (e.g. submission): just return logits
        outputs = self.backbone(**inputs)
        return outputs.logits


def get_tokenizer(model_name: str = None):
    """Return a tokenizer matching the corresponding backbone."""
    return AutoTokenizer.from_pretrained(model_name)
