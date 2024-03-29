from argparse import ArgumentParser


def get_my_args(parser: ArgumentParser):
    # Dataset Related Arguments.
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Absolute path to the datasets.",
    )

    parser.add_argument(
        "--unlabeled_dataset",
        required=True,
        type=str,
        help="Possible jsonl files.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Describes what dataset to use.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Directory to output the result.",
    )

    # Model Related Arguments.
    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        help="BERT Flavor to be used from HuggingFace.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=64,
        help="Maximum sequence length in terms of tokens for the input text",
    )
    parser.add_argument(
        "--text_soft_aug",
        type=str,
        default="eda_01",
        choices=["none", "eda", "eda_01", "eda_sr", "textDE"],
        help="Soft augmentation.",
    )
    parser.add_argument(
        "--text_hard_aug",
        type=str,
        default="eda_02",
        choices=["none", "eda", "eda_01", "eda_02", "textRU"],
        help="Strong augmentation.",
    )
    parser.add_argument(
        "--text_prob_aug",
        type=float,
        default=1.0,
        help="Probability of using augmented text.",
    )

    # Trainer Related Arguments.
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument(
        "--mu",
        default=1,
        type=int,
        help="Coefficient of unlabeled batch size. Unlabeled Batch Size = BS * Mu.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of workers to load data."
    )
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Gradient Accumulation Steps.",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--lrmain",
        "--learning-rate-main",
        default=1e-5,
        type=float,
        help="Initial learning rate for the BERT Layer from ClassificationBert.",
    )
    parser.add_argument(
        "--lrlast",
        "--learning-rate-last",
        default=1e-3,
        type=float,
        help="Initial learning rate for the Classifier Head from ClassificationBert.",
    )
    parser.add_argument(
        "--lr_factor", type=float, default=0.5, help="Learning rate factor."
    )
    parser.add_argument(
        "--lr_patience", type=int, default=2, help="Learning rate patience."
    )
    parser.add_argument(
        "--labeled_examples_per_class",
        type=int,
        default=0,
        help="How many labeled examples to use from each class. If 0 it uses all the samples.",
    )

    # Loss Related Arguments.
    parser.add_argument(
        "--lambda-u", default=1, type=float, help="Coefficient of unlabeled loss."
    )
    parser.add_argument(
        "--threshold", default=0.85, type=float, help="Pseudo label threshold."
    )
    parser.add_argument(
        "--linear_lu",
        action="store_true",
        default=False,
        help="Linearily increase lambda_u.",
    )
    parser.add_argument(
        "--random_lu",
        action="store_true",
        default=False,
        help="Randomly choose lambda_u for each batch.",
    )
    parser.add_argument(
        "--lambda-u-min",
        default=1.0,
        type=float,
        help="starting coefficient of unlabeled loss in case of linear schedule for it",
    )
    parser.add_argument(
        "--lambda-u-max",
        default=50.0,
        type=float,
        help="max coefficient of unlabeled loss in case of random schedule for it",
    )

    # Other Arguments.
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Only to evaluate the checkpoint given by args.resume",
    )
    parser.add_argument("--seed", default=666013, type=int, help="Seed.")
    parser.add_argument(
        "--no-progress", action="store_true", help="Don't use progress bar."
    )
