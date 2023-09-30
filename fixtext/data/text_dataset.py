import jsonlines
import random
from collections import Counter
import torch
from torch.utils.data import Dataset
from .vocab import Vocab
from pytorch_transformers import BertTokenizer
import argparse
from typing import Tuple, Union, Optional


class JsonlDataset(Dataset):
    """
    JSONL Dataset Class.

    Attributes:
        data (List[Dict[str, str]]): List of data samples.
        tokenizer (BertTokenizer): Tokenizer to be used on the text data.
        vocab (Vocab): Vocabulary object that holds the tokens & ids from tokenizer.
        n_classes (int): How many unique labels / classes are in the dataset.
        text_start_token (str): Start token for the input of BERT.
        pad_token (str): Pad token.
        max_seq_len (int): Maximum sequence length for the input text.
        text_aug_soft (str): Soft augmentation.
        text_aug (str): Strong augmentation.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizer,
        vocab: Vocab,
        n_classes: int,
        max_seq_len: int,
        labeled_examples_per_class: int = 0,
        text_aug_soft: str = "none",
        text_aug: Optional[str] = None,
    ):
        """
        Inits the JSONL Dataset Loader object.

        Args:
            data_path (str): Path to the JSONL file.
            tokenizer (BertTokenizer): Tokenizer used on input texts.
            vocab (Vocab): Vocabulary of the tokenizer.
            n_classes (int): How many unique labels / classes are in the dataset.
            max_seq_len (int): Maximum sequence length for the input text.
            labeled_examples_per_class (int): Number of samples to be used per class. This dataset loader
                tries to build a balanced dataset (as much as possible). If 0 it uses all the samples.
            text_aug_soft (str): Soft augmentation.
            text_aug (Optional[str]): Optional strong augmentation.
        """

        # Open the JSONL file and extract the objects.
        self.data = [obj for obj in jsonlines.open(data_path)]

        # Create the "backtranslation" augmentation from the English & German & Russian languages.
        for i in range(len(self.data)):
            self.data[i]["backtranslation"] = [
                self.data[i]["text"],
                self.data[i]["textDE"],
                self.data[i]["textRU"],
            ]
            self.data[i]["textDE"] = [self.data[i]["textDE"]]
            self.data[i]["textRU"] = [self.data[i]["textRU"]]

        # Extract the labels.
        labels = {int(x["label"]) for x in self.data}

        # Extracts a concrete number of samples.
        if labeled_examples_per_class > 0:
            data = []

            # For each label extract a number of samples at random.
            # The number of samples to be extracted is given by the "labeled_examples_per_class" parameter.
            for label in labels:
                label_data = [x for x in self.data if int(x["label"]) == label]
                label_data = random.sample(label_data, labeled_examples_per_class)
                data += label_data

            # Shuffle the extracted data.
            random.shuffle(data)
            self.data = data

        # Print the statistic regarding the number of samples per class.
        print(f'Final data split as: {Counter([int(x["label"]) for x in self.data])}')
        print()

        # Save the arguments.
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.n_classes = n_classes
        self.text_start_token = ["[CLS]"]
        self.pad_token = ["[PAD]"]

        self.max_seq_len = max_seq_len

        self.text_aug_soft = text_aug_soft
        self.text_aug = text_aug

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            (int): The length of the dataset.
        """

        return len(self.data)

    def get_sentence_and_segment(self, text: str) -> torch.LongTensor:
        """
        Returns the token ids for the given text. The text is firstly tokenized and then
        we use the vocabulary property to extract the ids corresponding to the tokens.

        Args:
            text (str): Text to be tokenized.

        Returns:
            (torch.LongTensor): Tensor with the token ids for the input text.
        """

        # First tokenize the text. The text will be truncated to "max_seq_len".
        tokenized_text = self.tokenizer(text)[: (self.max_seq_len - 1)]

        # Add the Start Token and pad the input text to "max_seq_len".
        sentence = (
            self.text_start_token
            + tokenized_text
            + self.pad_token * (self.max_seq_len - len(tokenized_text) - 1)
        )

        # Extract ids for the tokens.
        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        return sentence

    def _get_text(self, index: int, aug: str) -> torch.LongTensor:
        """
        For a given sample (specified by the "index" parameter) we apply a given
        augmentation (specified by the "aug" parameter). We return the ids corresponding to the tokenized augmented text.

        Args:
            index (int): Index of the sample.
            aug (str): What augmentation to apply.

        Returns:
            (torch.LongTensor): Tensor with the token ids for the input text.
        """

        if aug == "none":
            text = self.data[index]["text"]
        else:
            # Each augmentation has a precomputed list of values.
            # Extract at random one possible value for this augmentation.
            text = random.choice(self.data[index][aug])

        # Extract the ids for the given augmented sample.
        sentence = self.get_sentence_and_segment(text)
        return sentence

    def get_item(
        self, index: int
    ) -> Tuple[torch.LongTensor, int, Union[torch.LongTensor, None]]:
        """
        Returns a softly augmented version of a sample, its label and optionally
        a strongly augmented version of the respective sample.

        Args:
            index (int): The sample we need to return.

        Returns:
            (Tuple[torch.LongTensor, int, torch.LongTensor | None]):
                The softly augmented sample, the label, and optionally, the strongly augmented sample.
        """

        # Softly augment the text & extract the ids corresponding to the augmented tokenized sample.
        sentence = self._get_text(index, self.text_aug_soft)

        # Extract the label.
        label = int(self.data[index]["label"])

        # Check if we need to strongly augment the sample.
        if self.text_aug is None:
            return sentence, label, None
        else:
            # Strongly augment the sample & extract the ids.
            sentence_aug = self._get_text(index, self.text_aug)
            return sentence, label, sentence_aug

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.LongTensor, int, Union[torch.LongTensor, None]]:
        """
        Get a sample from the dataset.

        Args:
            index (int): The sample we need to return.

        Returns:
            (Tuple[torch.LongTensor, int, torch.LongTensor | None]):
                The softly augmented sample, the label, and optionally, the strongly augmented sample.
        """

        return self.get_item(index)
