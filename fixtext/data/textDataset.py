import jsonlines
import json
import numpy as np
import os
from PIL import Image
import random
from collections import Counter
import torch
from torch.utils.data import Dataset
from vocab import Vocab

from utils.utils import truncate_seq_pair, numpy_seed

class JsonlDataset(Dataset):
    '''
    args.labels
    args.max_seq_len
    '''
    def __init__(
        self,
        data_path: str,
        tokenizer: any,
        vocab: Vocab,
        args,
        labeled_examples_per_class: int = 0,
        text_aug0='none',
        text_aug=None
    ):
    
        self.data = [obj for obj in jsonlines.open(data_path)]
        
        labels = { int(x['label']) for x in self.data }
        
        # Extracts a concrete number of samples.
        if labeled_examples_per_class > 0:
            data = []
            for label in labels:
                label_data = [x for x in self.data if int(x['label']) == label]
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

        self.max_seq_len = args.max_seq_len

        self.errors = 0
        self.text_aug0 = text_aug0
        self.text_aug = text_aug

            
    def __len__(self):
        return len(self.data)

    def get_sentence_and_segment(self, text: str):
        sentence = (
            self.text_start_token
            + self.tokenizer(text)[
                : (self.args.max_seq_len - 1)
            ]
        )
        segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        return sentence, segment

    def _get_text(self, index, aug):
        if aug == 'none':
            text = self.data[index]["text"]
        else:
            text = random.choice(self.data[index][aug])

        sentence, segment = self.get_sentence_and_segment(text)
        return sentence, segment


    def get_item(self, index):
        sentence, segment = self._get_text(index, self.text_aug0)

        label = int(self.data[index]["label"])
        
        if self.text_aug is None:
            return sentence, segment, label
        else:
            sentence_aug, segment_aug = self._get_text(index, self.text_aug)
            return sentence, segment, label, sentence_aug, segment_aug
        
    
    def __getitem__(self, index):
        return self.get_item(index)
