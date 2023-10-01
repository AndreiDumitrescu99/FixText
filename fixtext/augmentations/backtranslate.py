import torch
from typing import Tuple
import jsonlines
from fixtext.augmentations.eda import get_only_chars


def load_backtranslation_model(language: str) -> Tuple[any, any]:
    """
    Loads model from fairseq for backtranslation. Supports en-de and en-ru as language backtranslations.

    Args:
        language (str): Language for which we load the backtranslation models. Supports either 'de' or 'ru'.

    Returns:
        (Tuple[any, any]): The forward and backward translating models.
    """

    if language == "ru":
        model_forward = torch.hub.load(
            "pytorch/fairseq",
            "transformer.wmt19.en-ru.single_model",
            tokenizer="moses",
            bpe="fastbpe",
        )
        model_backward = torch.hub.load(
            "pytorch/fairseq",
            "transformer.wmt19.ru-en.single_model",
            tokenizer="moses",
            bpe="fastbpe",
        )
    elif language == "de":
        model_forward = torch.hub.load(
            "pytorch/fairseq",
            "transformer.wmt19.en-de.single_model",
            tokenizer="moses",
            bpe="fastbpe",
        )
        model_backward = torch.hub.load(
            "pytorch/fairseq",
            "transformer.wmt19.de-en.single_model",
            tokenizer="moses",
            bpe="fastbpe",
        )
    else:
        raise ValueError(
            "Only languages supported, for the moment, are: 'ru' and 'de'."
        )

    return model_forward, model_backward


def backtranslate(text: str, model_forward: any, model_backward: any) -> str:
    """
    Backtranslates text from one language to another.

    Args:
        text (str): Text to backtranslate.
        model_forward (any): Model to translate the text from the original language to an intermediary one.
        model_backward (any): Model to translate the text from the intermediary language to an original one.

    Returns:
        (str): Backtranslated text.
    """

    return model_backward.translate(
        model_forward.translate(text, sampling=True, temperature=0.9),
        sampling=True,
        temperature=0.9,
    )


def precompute_backtranslate(
    model_forward: any,
    model_backward: any,
    input_file: str,
    output_file: str,
    language: str,
):
    """
    Precomputes the backtranslation augmentation. Saves the samples to a new jsonl file.

    Args:
        model_forward (any): Model to translate the text from the original language to an intermediary one.
        model_backward (any): Model to translate the text from the intermediary language to an original one.
        input_file (str): JSONL file with the samples.
        output_file (str): JSONL output file with the samples.
        language (str): Intermediary language used for backtranslation, either 'ru' or 'de'.
    """

    with jsonlines.open(input_file, mode="r") as reader, jsonlines.open(
        file=output_file, mode="w"
    ) as writer:
        for _, data in enumerate(reader):
            data["text"] = get_only_chars(data["text"])
            text_backtranslated = backtranslate(
                data["text"], model_forward, model_backward
            )

            data[f"text{language.upper()}"] = text_backtranslated

            writer.write(data)


if __name__ == "__main__":
    model_forward, model_backward = load_backtranslation_model("ru")

    precompute_backtranslate(
        model_forward,
        model_backward,
        "../../data_example/initial_json_de.jsonl",
        "../../data_example/initial_json_backtranslation.jsonl",
        "ru",
    )
