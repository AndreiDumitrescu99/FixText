from typing import List


class Vocab(object):
    """
    Vocabulary Class that holds a mapping between words and ids.
    """

    def __init__(self, emptyInit: bool = False):
        """
        Inits the Vocabulary Object. It holds 2 mappings: a mapping from string to an id and
        another reverse mapping from id to string. We also store the vocabulary size.

        Args:
            emptyInit (bool): Boolean flag that tells us if we make an empty init or not.
                If 'False', the mapping will be initialized with the special tokens from
                the BERT Tokenizers (e.g.: [CLS], [SEP] etc.).
        """

        if emptyInit:
            self.stoi, self.itos, self.vocab_sz = {}, [], 0
        else:
            # Init the mappings with the special tokens.
            self.stoi = {
                w: i
                for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
            }
            self.itos = [w for w in self.stoi]
            self.vocab_sz = len(self.itos)

    def add(self, words: List[str]) -> None:
        """
        Add words to the vocabulary.

        Args:
            words (List[str]): List of words to be added to the vocabulary.
        """

        # Extract the current id.
        cnt = len(self.itos)
        # For each word check if it is already in the mappings. If it isn't, add it.
        for w in words:
            if w in self.stoi:
                continue
            self.stoi[w] = cnt
            self.itos.append(w)
            cnt += 1
        # Save the new vocabulary size.
        self.vocab_sz = len(self.itos)
