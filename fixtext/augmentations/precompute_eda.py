"""
Scripts for precomputing eda.
"""
from eda import eda, get_only_chars
from argparse import ArgumentParser, Namespace
import os
import jsonlines
from tqdm import tqdm
from utils.utils import set_seed
from typing import List


def sanity_checks(args: Namespace):
    """
    Applies sanity checks, meaning that the configuration provided through argparse is correct.

    Args:
        (argparse.Namespace): Arguments for which we run the sanity checks.
    """

    # Make sure we have the same number of input and output files.
    assert len(args.input_files) == len(args.output_files)

    for input_file, output_file in zip(args.input_files, args.output_files):
        # Make sure that the input files exist.
        assert os.path.isfile(input_file), f"{input_file} DOES NOT EXIST!"

        # If we don't want to override the output files, make sure they don't already exist.
        assert args.override or not os.path.isfile(
            output_file
        ), f"{output_file} ALREADY EXISTS!"


def get_args_for_eda() -> Namespace:
    """
    Get necessary arguments for precomputing EDA Augmentations.

    Returns:
        (argparse.Namespace): The arguments for precomputing EDA augmentations.
    """

    parser = ArgumentParser()

    parser.add_argument(
        "--input_files",
        type=List[str],
        required=True,
        action="append",
        help="List of input files with the samples that need to be augmented. The files should be jsonl files.",
    )

    parser.add_argument(
        "--output_files",
        type=List[str],
        required=True,
        action="append",
        help="List of output files where we save the augmentated samples.",
    )

    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=10,
        help="Number of operations to apply per sample for each augmentation.",
    )

    parser.add_argument(
        "--override",
        action="store_true",
        help="If we want to override existing output files.",
    )

    args = parser.parse_args()

    # Run sanity checks.
    sanity_checks(args)

    return args


def main(
    input_files: List[str],
    output_files: List[str],
    num_augmentations: int,
):
    """
    Main function to precompute EDA. It computes: EDA-01, EDA-02 and EDA-SR.
    Each input file will have a corresponding output file, where the function will
    log the samples completed with the new augmentations.

    Args:
        input_files (List[str]): List of input files.
        output_files (List[str]): List of output files.
        num_augmentations (int): How many augmentations to compute for each sample.
    """

    # Set seed for reproductibility.
    set_seed(666013)

    for input_file, output_file in zip(input_files, output_files):
        # Determine the full paths.
        input_file, output_file = os.path.join(input_file), os.path.join(output_file)

        # Create the directory for the ourput file (if it doesn't exist).
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print(f"\n Adding EDA to input file {input_file}")

        # Open the input and output files. The files should be jsonl files.
        with jsonlines.open(file=input_file, mode="r") as reader, jsonlines.open(
            file=output_file, mode="w"
        ) as writer:
            # Take each (JSON) sample from the input file and compute the augmentations.
            for _, obj in tqdm(enumerate(reader)):
                try:
                    # Apply EDA-01 (alpha_sr = alpha_ri = alpha_rs = p_rd = 0.1)
                    obj["eda_01"] = eda(
                        obj["text"], 0.1, 0.1, 0.1, 0.1, num_augmentations
                    )

                    # Apply EDA-SR (alpha_sr = 0.1 & alpha_ri = alpha_rs = p_rd = 0.0)
                    obj["eda_sr"] = eda(
                        obj["text"], 0.1, 0.0, 0.0, 0.0, num_augmentations, True
                    )

                    # Apply EDA-02 (alpha_sr = alpha_ri = alpha_rs = p_rd = 0.2)
                    obj["eda_02"] = eda(
                        obj["text"], 0.2, 0.2, 0.2, 0.2, num_augmentations
                    )

                    # Clean the Text and the Backtranslations.
                    obj["text"] = get_only_chars(obj["text"])
                    obj["textDE"] = get_only_chars(obj["textDE"])
                    obj["textRU"] = get_only_chars(obj["textRU"])

                except Exception as excpt:
                    print(excpt)
                    print(f'Error at sample with text: {obj["text"]}.')
                    continue

                writer.write(obj)


if __name__ == "__main__":
    args = get_args_for_eda()
    main(args.input_files, args.output_files, args.num_augmentations)
