from eda import eda, get_only_chars
from argparse import ArgumentParser, Namespace
import os
import jsonlines
from tqdm import tqdm
from utils.utils import set_seed
from typing import List


def sanity_checks(args):
    assert len(args.input_file) == len(args.output_file)
    for input_file, output_file in zip(args.input_file, args.output_file):
        assert os.path.isfile(input_file), f"{input_file} DOES NOT EXIST!"
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
        "--input_file",
        type=List[str],
        required=True,
        action="append",
        help="List of input files with the samples that need to be augmented.",
    )

    parser.add_argument(
        "--output_file",
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

    return args


def main(args: Namespace):
    set_seed(666013)

    sanity_checks(args)

    for input_file, output_file in zip(args.input_file, args.output_file):
        input_file, output_file = os.path.join(input_file), os.path.join(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"\n Adding EDA to input file {input_file}")

        # Open the input and output files.
        with jsonlines.open(input_file) as reader, jsonlines.open(
            output_file, "w"
        ) as writer:
            for _, obj in tqdm(enumerate(reader)):
                try:
                    # Apply EDA-01 (alpha_sr = alpha_ri = alpha_rs = p_rd = 0.1)
                    obj["eda_01"] = eda(
                        obj["text"], 0.1, 0.1, 0.1, 0.1, args.num_augmentations
                    )

                    # Apply EDA-SR (alpha_sr = 0.1 & alpha_ri = alpha_rs = p_rd = 0.0)
                    obj["eda_sr"] = eda(
                        obj["text"], 0.1, 0.0, 0.0, 0.0, args.num_augmentations, True
                    )

                    # Apply EDA-02 (alpha_sr = alpha_ri = alpha_rs = p_rd = 0.2)
                    obj["eda_02"] = eda(
                        obj["text"], 0.2, 0.2, 0.2, 0.2, args.num_augmentations
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
    main(get_args_for_eda())
