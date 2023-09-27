import jsonlines
import os
from collections import Counter, namedtuple

import torch
import torchvision.transforms as transforms
from pytorch_transformers import *

from .text_dataset import JsonlDataset
from data.vocab import Vocab

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class Object(object):
    pass

def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [int(obj['label']) for obj in jsonlines.open(path)]

    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    print(f'Labels frequencies: {label_freqs}')

    return list(label_freqs.keys()), label_freqs

def get_vocab(args):
    vocab = Vocab()
    bert_tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    )
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)

    return vocab

def get_datasets(args):
    
    tokenizer = (
        BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    )
    
    args.labels, args.label_freqs = get_labels_and_frequencies(os.path.join(args.data_path, 'train', args.task))

    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)
    print('args.labels:', args.labels)
    print('args.label_freqs', args.label_freqs)
    print('n_classes:', args.n_classes)

    labeled_examples_per_class = args.labeled_examples_per_class #if args.unbalanced else num_expand_x // args.n_classes

    labeled_set = JsonlDataset(
        os.path.join(args.data_path, 'train', args.task),
        tokenizer,
        vocab,
        args,
        labeled_examples_per_class=labeled_examples_per_class,
        text_aug0=args.text_soft_aug
    )

    args.train_data_len = len(labeled_set)

    unlabeled_set = JsonlDataset(
        os.path.join(args.data_path, 'unlabeled', args.unlabeled_dataset),
        tokenizer,
        vocab,
        args,
        text_aug0=args.text_soft_aug,
        text_aug=args.text_hard_aug
    )
    
    dev_set = JsonlDataset(
        os.path.join(args.data_path, 'dev', args.task),
        tokenizer,
        vocab,
        args,
        text_aug0='none'
    )
    
    test_set = JsonlDataset(
        os.path.join(args.data_path, 'test', args.task),
        tokenizer,
        vocab,
        args,
        text_aug0='none'
    )
    
    return labeled_set, unlabeled_set, dev_set, test_set

def get_data_loaders(args):

    labeled_set, unlabeled_set, dev_set, test_set = get_datasets(args)
        
    labeled_loader = DataLoader(
        labeled_set,
        sampler = RandomSampler(labeled_set),
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        drop_last = True
    )

    unlabeled_loader = DataLoader(
        unlabeled_set,
        sampler = RandomSampler(unlabeled_set),
        batch_size = int(args.batch_size * args.mu),
        num_workers = args.num_workers,
        drop_last = True
    )
    
    dev_loader = DataLoader(
        dev_set,
        sampler = SequentialSampler(dev_set),
        batch_size = args.batch_size,
        num_workers = args.num_workers
    ) 

    test_loader = DataLoader(
        test_set,
        sampler = SequentialSampler(test_set),
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )

    return labeled_loader, unlabeled_loader, dev_loader, test_loader

if __name__ == '__main__':

    args = Object()

    args.data_path = 'C:\\Users\\andre\\Desktop\\Master - 2\\Research\\final_datasets\\fixtext\\'
    args.task = 'final_simplified_language_youtube_parsed_dataset.jsonl'
    args.unlabeled_dataset = 'unlabled_language_youtube_dataset.jsonl'
    args.batch_size = 1
    args.num_workers = 1
    args.mu = 1
    args.bert_model = 'bert-base-uncased'
    args.text_soft_aug = 'eda_01'
    args.text_hard_aug = 'textRU'
    args.max_seq_len = 60

    labeled_loader, unlabeled_loader, dev_loader, test_loader = get_data_loaders(args)

    train_loader = zip(labeled_loader, unlabeled_loader)

    print("Loaded!")

    for batch_idx, (data_x, data_u) in enumerate(train_loader):

        print("Labeled: ", data_x)
        print()
        print("Unlabeled:", data_u)

        break

