# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
from typing import List

import re
import nltk
from nltk.corpus import wordnet
from .stop_words import stop_words

nltk.download("wordnet")
nltk.download("omw-1.4")
random.seed(1)


def reduce_repeating(s: str) -> str:
    repeat_pattern = re.compile(r"(.)\1\1+")
    match_substitution = r"\1\1"
    return repeat_pattern.sub(match_substitution, s)


def get_only_chars(line: str) -> str:
    """
    Receives a string line and returns the cleaned version of it which contains only alphabet characters.

    Args:
        line (str): String line to be cleaned.

    Returns:
        (str): Cleaned line containing only lower cased alphabet characters.
    """

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  # Replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in "0123456789qwertyuiopasdfghjklzxcvbnm ":
            clean_line += char
        else:
            clean_line += " "

    # Delete extra spaces.
    clean_line = re.sub(" +", " ", clean_line)
    if clean_line[0] == " ":
        clean_line = clean_line[1:]

    # Reduce repeaitng characters.
    clean_line = reduce_repeating(clean_line)

    return clean_line


def get_synonyms(word: str) -> List[str]:
    """
    Get list of synonims for a given word.

    Args:
        word (str): Word for which we want to compute synonims.

    Returns:
        (List[str]): List of synonims for the given words.
    """

    synonyms = set()

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join(
                [char for char in synonym if char in " qwertyuiopasdfghjklzxcvbnm"]
            )
            synonyms.add(synonym)

    if word in synonyms:
        synonyms.remove(word)

    return list(synonyms)


def synonym_replacement(words: List[str], n: int) -> List[str]:
    """
    Replace N words from a given list with their synonims.

    Args:
        words (List[str]): List of words.
        n (int): How many words to replace.

    Returns:
        (List[str]): List with the new words replaced.
    """

    new_words = words.copy()

    # Remove stop words.
    random_word_list = list(set([word for word in words if word not in stop_words]))

    # Shuffle the list of words.
    random.shuffle(random_word_list)

    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)

        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1

        # Only replace up to n words.
        if num_replaced >= n:
            break

    # This is stupid but we need it, trust me. | A.D.: Yeah, it is.
    sentence = " ".join(new_words)
    new_words = sentence.split(" ")

    return new_words


def random_deletion(words: List[str], p: float) -> List[str]:
    """
    Delete at random words from a list. A word has a chance of p to be deleted.

    Args:
        words (List[str]): List of words to be deleted.
        p (float): Probability that defines the chance of a word to be deleted.

    Returns:
        (List[str]): New list of words without the deleted ones.
    """

    # Obviously, if there's only one word, don't delete it.
    if len(words) == 1:
        return words

    # Randomly delete words with probability p.
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # If you end up deleting all words, just return a random word.
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


def swap_word(new_words: List[str]) -> List[str]:
    """
    Swap 2 words from a given list of words. It has a small chance to return the list unchanged.

    Args:
        new_words (List[str]): List of initial words.

    Returns:
        (List[str]): The list with 2 words swapped, or in a very small of cases the list unchanged.
    """

    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1

    # Try to get 2 different random indexes for 3 times.
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words

    # Swap the words.
    new_words[random_idx_1], new_words[random_idx_2] = (
        new_words[random_idx_2],
        new_words[random_idx_1],
    )

    return new_words


def random_swap(words: List[str], n: int) -> List[str]:
    """
    Perform N word swaps.

    Args:
        words (List[str]): List of words on which we want to perform the swaps.
        n (int): The number of swaps to be done.

    Returns:
        (List[str]): List of swapped words.
    """

    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def add_word(new_words: List[str]) -> None:
    """
    Insert a word inside of the list of words. The new word has to be a synonim of a random word from the list.
    The synonim is added at a random position. If the model can't find in 10 tries a synonim it returns the original list.

    Args:
        new_words (List[str]): List of original words.

    Returns:
        (None): The list of words is changed in place!
    """

    synonyms = []
    counter = 0

    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return

    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


def random_insertion(words: List[str], n: int) -> List[str]:
    """
    Add N words inside the word list.

    Args:
        words (List[str]): Original list of words.
        n (int): Number of insertions.

    Returns:
        (List(str)): New list of words.
    """

    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


########################################################################
# main data augmentation function
########################################################################


def eda(
    sentence: str,
    alpha_sr: float = 0.1,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    p_rd: float = 0.1,
    num_aug: int = 10,
    per_technique: bool = False,
) -> List[str]:
    # Preproces the sentence.
    sentence = get_only_chars(sentence)
    words = sentence.split(" ")
    words = [word for word in words if word != ""]
    num_words = len(words)

    augmented_sentences = []
    if per_technique:
        num_new_per_technique = num_aug
        num_aug = 4 * num_new_per_technique
    else:
        num_new_per_technique = int(num_aug / 4) + 1

    # Synonym replacement.
    if alpha_sr > 0:
        n_sr = max(1, int(alpha_sr * num_words))
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(" ".join(a_words))

    # Random Insertion.
    if alpha_ri > 0:
        n_ri = max(1, int(alpha_ri * num_words))
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.append(" ".join(a_words))

    # Random Swap.
    if alpha_rs > 0:
        n_rs = max(1, int(alpha_rs * num_words))
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(" ".join(a_words))

    # Random Deletion.
    if p_rd > 0:
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, p_rd)
            augmented_sentences.append(" ".join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    # Trim so that we have the desired number of augmented sentences.
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [
            s for s in augmented_sentences if random.uniform(0, 1) < keep_prob
        ]

    # Append the original sentence.
    augmented_sentences.append(sentence)

    return augmented_sentences
