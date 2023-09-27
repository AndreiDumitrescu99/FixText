import torch
import torch.nn as nn
from pytorch_transformers import *


class ClassificationBert(nn.Module):
    """
    Classification BERT, main architecture used for the training of FixText. It is composed of a
    BERT Base Uncased Layer and a Classification Head built on top of the BERT Layer. The model receives as
    input the tokens for the text input and returns the probabilities for each of the possible classes.

    The BERT Base Uncased will be used as a feature extractor. It returns the contextual embeddings for each
    token. We average these embeddings to form an overall context embedding. This overall context embedding
    is passed to the Classification Head.

    The Classification Head is made out of a Linear Layer used to reduce the embeddings dimensionality, followed by a
    Tanh activation function and a last Linear Layer that reduced the dimensionality of the embeddings
    to the number of classes used for classification.
    """

    def __init__(self, num_labels: int = 2):
        """
        Inits the Classification BERT.

        Args:
            num_labels (int): Number of classes. Used to initialize the Classification Head. Defaults to 2.
        """

        super(ClassificationBert, self).__init__()

        # Load pre-trained BERT model.
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # Build the Classification Head.
        self.linear = nn.Sequential(
            nn.Linear(768, 128), nn.Tanh(), nn.Linear(128, num_labels)
        )

    def forward(self, x: torch.Tensor, length: int = 256) -> torch.Tensor:
        """
        Forward pass through the Classification BERT.

        Args:
            x (torch.Tensor): The input text tokens.
            length (int): The maximum number of text tokens from the sample.

        Returns:
            (torch.Tensor): The output of the model.
        """

        # Encode the input text using the pre-trained BERT model.
        all_hidden, _ = self.bert(x)

        # Average the embeddings to obtain the context embedding.
        pooled_output = torch.mean(all_hidden, 1)

        # Use Classification Head to obtain the final predictions.
        predict = self.linear(pooled_output)

        return predict
