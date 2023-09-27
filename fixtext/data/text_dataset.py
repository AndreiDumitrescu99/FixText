import jsonlines
import random
from collections import Counter
import torch
from torch.utils.data import Dataset
from .vocab import Vocab


class JsonlDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: any,
        vocab: Vocab,
        args,
        labeled_examples_per_class: int = 0,
        text_aug0: str = "none",
        text_aug=None,
    ):
        self.data = [obj for obj in jsonlines.open(data_path)]

        for i in range(len(self.data)):
            self.data[i]["backtranslation"] = [
                self.data[i]["text"],
                self.data[i]["textDE"],
                self.data[i]["textRU"],
            ]
            self.data[i]["textDE"] = [self.data[i]["textDE"]]
            self.data[i]["textRU"] = [self.data[i]["textRU"]]

        labels = {int(x["label"]) for x in self.data}

        # Extracts a concrete number of samples.
        if labeled_examples_per_class > 0:
            data = []
            for label in labels:
                label_data = [x for x in self.data if int(x["label"]) == label]
                label_data = random.sample(label_data, labeled_examples_per_class)
                data += label_data
            random.shuffle(data)
            self.data = data

        print(f'Final data split as: {Counter([int(x["label"]) for x in self.data])}')
        print()

        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"]
        self.pad_token = ["[PAD]"]

        self.max_seq_len = args.max_seq_len

        self.errors = 0
        self.text_aug0 = text_aug0
        self.text_aug = text_aug

    def __len__(self):
        return len(self.data)

    def get_sentence_and_segment(self, text: str):
        tokenized_text = self.tokenizer(text)[: (self.args.max_seq_len - 1)]
        sentence = (
            self.text_start_token
            + tokenized_text
            + self.pad_token * (self.max_seq_len - len(tokenized_text) - 1)
        )

        # segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        return sentence

    def _get_text(self, index, aug):
        if aug == "none":
            text = self.data[index]["text"]
        else:
            text = random.choice(self.data[index][aug])

        sentence = self.get_sentence_and_segment(text)
        return sentence

    def get_item(self, index):
        sentence = self._get_text(index, self.text_aug0)

        label = int(self.data[index]["label"])

        if self.text_aug is None:
            return sentence, label
        else:
            sentence_aug = self._get_text(index, self.text_aug)
            return sentence, label, sentence_aug

    def __getitem__(self, index):
        return self.get_item(index)
