import functools
import jsonlines
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from pytorch_transformers import *

from textDataset import JsonlDataset
from data.vocab import Vocab

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [int(obj["label"]) for obj in jsonlines.open(path)]

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

def get_datasets(args, num_expand_x, num_expand_u):
    
    tokenizer = (
        BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    )
    
    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, f"{args.train_file}.jsonl")
    )

    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)
    print('args.labels:', args.labels)
    print('args.label_freqs', args.label_freqs)
    print('n_classes:', args.n_classes)

    labeled_examples_per_class = 0 if args.unbalanced else num_expand_x // args.n_classes

    labeled_set = JsonlDataset(
        os.path.join(args.data_path, args.task, f"{args.train_file}.jsonl"),
        tokenizer,
        vocab,
        args,
        labeled_examples_per_class=labeled_examples_per_class,
        text_aug0=args.text_soft_aug
    )

    args.train_data_len = len(labeled_set)

    unlabeled_set = JsonlDataset(
        os.path.join(args.data_path, 'unlabeled', f"{args.unlabeled_dataset}.jsonl"),
        tokenizer,
        vocab,
        args,
        text_aug0=args.text_soft_aug,
        text_aug=args.text_hard_aug
    )
    
    dev_set = JsonlDataset(
        os.path.join(args.data_path, args.task, "dev.jsonl"),
        tokenizer,
        vocab,
        args,
        text_aug0='none'
    )
    
    test_set = JsonlDataset(
        os.path.join(args.data_path, args.task, "test.jsonl"),
        tokenizer,
        vocab,
        args,
        text_aug0='none'
    )
    
    return labeled_set, unlabeled_set, dev_set, test_set

def get_data_loaders(args):

    labeled_set, unlabeled_set, dev_set, test_set = get_datasets(args, args.k_img, args.k_img*args.mu)
        
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
        batch_size = args.batch_size * args.mu,
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