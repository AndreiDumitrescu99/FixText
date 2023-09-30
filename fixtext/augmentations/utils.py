""" Utils function for computing augmentations """
import re


def reduce_repeating(s: str) -> str:
    """
    Reduce repeating characters.

    Args:
        s (str): String which we want to clean.

    Returns:
        (str): The cleaned string.
    """

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
