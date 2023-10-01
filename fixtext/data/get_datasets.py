from argparse import Namespace
from collections import Counter
import jsonlines
import os

from pytorch_transformers import *

from fixtext.data.text_dataset import JsonlDataset
from fixtext.data.vocab import Vocab

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from typing import Tuple, List


class Object(object):
    pass


def get_labels_and_frequencies(path: str) -> Tuple[List[int], Counter]:
    """
    Extract a list of labels from the dataset and also the labels' frequencies.

    Args:
        path (str): Path to a jsonl file with the dataset.

    Returns:
        (Tuple[List[int], Counter]): List of unique labels and a counter with their frequencies.
    """

    label_freqs = Counter()
    data_labels = [int(obj["label"]) for obj in jsonlines.open(path)]

    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    print(f"Labels frequencies: {label_freqs}")

    return list(label_freqs.keys()), label_freqs


def get_vocab(bert_model: str) -> Vocab:
    """
    Function that inits a Vocab object with the tokens and ids from the BertTokenizer.

    Args:
        bert_model (str): The flavor of BERT for the BertTokenizer.

    Returns:
        (Vocab): Vocabulary object initialized with the tokens and ids from BertTokenzier.
    """

    vocab = Vocab()
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)

    return vocab


def get_datasets(
    args: Namespace,
) -> Tuple[JsonlDataset, JsonlDataset, JsonlDataset, JsonlDataset]:
    """
    Function that inits the 4 JsonlDataset Objects: one for the labeled training set,
    one for the unlabeled training part, one for the dev set and one for the test set.

    Args:
        args (argparse.Namespace): Object with all the details needed to initialize the datasets.

    Returns:
        (Tuple[JsonlDataset, JsonlDataset, JsonlDataset, JsonlDataset]): The 4 JsonlDatasets.
    """

    # Load the tokenizer.
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    ).tokenize

    # Get the unique labels from the dataset & their frequencies.
    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, "train", args.task)
    )

    # Load the vocab for the tokenizer.
    vocab = get_vocab(args.bert_model)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    labeled_examples_per_class = args.labeled_examples_per_class

    # Load the JsonlDataset for the labeled training dataset.
    labeled_set = JsonlDataset(
        data_path=os.path.join(args.data_path, "train", args.task),
        tokenizer=tokenizer,
        vocab=vocab,
        n_classes=len(args.labels),
        max_seq_len=args.max_seq_len,
        labeled_examples_per_class=labeled_examples_per_class,
        text_aug_soft=args.text_soft_aug,
    )

    args.train_data_len = len(labeled_set)

    # Load the JsonlDataset for the unlabeled training dataset.
    unlabeled_set = JsonlDataset(
        data_path=os.path.join(args.data_path, "unlabeled", args.unlabeled_dataset),
        tokenizer=tokenizer,
        vocab=vocab,
        n_classes=len(args.labels),
        max_seq_len=args.max_seq_len,
        text_aug_soft=args.text_soft_aug,
        text_aug=args.text_hard_aug,
    )

    # Load the JsonlDataset for the validation dataset.
    dev_set = JsonlDataset(
        data_path=os.path.join(args.data_path, "dev", args.task),
        tokenizer=tokenizer,
        vocab=vocab,
        n_classes=len(args.labels),
        max_seq_len=args.max_seq_len,
        text_aug_soft="none",
    )

    # Load the JsonlDataset for the testing dataset.
    test_set = JsonlDataset(
        data_path=os.path.join(args.data_path, "test", args.task),
        tokenizer=tokenizer,
        vocab=vocab,
        n_classes=len(args.labels),
        max_seq_len=args.max_seq_len,
        text_aug_soft="none",
    )

    return labeled_set, unlabeled_set, dev_set, test_set


def get_data_loaders(
    args: Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Build 4 Dataset Loaders used during the training process: one for the labeled training set,
    one for the unlabeled training part, one for the dev set and one for the test set.

    Args:
        args (argparse.Namespace): Object with all the details needed to initialize the datasets.

    Returns:
        (Tuple[DataLoader, DataLoader, DataLoader, DataLoader]): The 4 DataLoader.
    """

    # Extract the 4 JsonlDatasets.
    labeled_set, unlabeled_set, dev_set, test_set = get_datasets(args)

    # Build the DataLoader for the labeled training set.
    labeled_loader = DataLoader(
        labeled_set,
        sampler=RandomSampler(labeled_set),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # Build the DataLoader for the unlabeled training set.
    unlabeled_loader = DataLoader(
        unlabeled_set,
        sampler=RandomSampler(unlabeled_set),
        batch_size=int(args.batch_size * args.mu),
        num_workers=args.num_workers,
        drop_last=True,
    )

    # Build the DataLoader for the validation set.
    dev_loader = DataLoader(
        dev_set,
        sampler=SequentialSampler(dev_set),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Build the DataLoader for the testing set.
    test_loader = DataLoader(
        test_set,
        sampler=SequentialSampler(test_set),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return labeled_loader, unlabeled_loader, dev_loader, test_loader
