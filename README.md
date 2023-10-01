# Fix Text
Fix Text extends the idea behind of [Fix Match](https://arxiv.org/ftp/arxiv/papers/2001/2001.07685.pdf) to text based inputs. The proposed method also builds upon the work done in the ["Multimodal Semi-supervised Learning for Disaster Tweet Classification”](https://aclanthology.org/2022.coling-1.239.pdf) research paper by extracting the main ideas behind the text augmentations. The algorithm for Fix Text is really similar to the Fix Match one, the only main difference being the augmentations we apply on the input text. This code was used to produce the results presented in this [paper](https://drive.google.com/file/d/1UxUAsKJvt7pgwdc7eqLMTxTx6SgjR2Or/view?usp=sharing).

## Our Approach - Cyberbullying Detection Task
Our goal is to to compare how models trained in a semi supervised manner using the Fix Text algorithm perform against the models trained in
a classic supervised way on a concrete task. <br>
Fix Text was tested on the Cyberbullying Detection task on 4 different datasets, in 2 separate scenarios. Each dataset has a different data distribution from the others, and they try to cover distinct social media contexts: Tweets, YouTube Comments and Kaggle Comments. When training the model in a semi supervised way on one of the 4 datasets will act as a labeled dataset, the rest 3 of them will act as an unlabeled dataset. The datasets used are: the [Cyberbullying Tweets dataset](https://people.cs.vt.edu/ctlu/Publication/2020/IEEE-BD-SOSNet-Wang.pdf), [Kaggle Comments dataset ](https://www.researchgate.net/publication/353392781_When_the_Timeline_Meets_the_Pipeline_A_Survey_on_Automated_Cyberbullying_Detection), [YouTube Comments dataset](https://www.researchgate.net/publication/353392781_When_the_Timeline_Meets_the_Pipeline_A_Survey_on_Automated_Cyberbullying_Detection) and another [Tweets Dataset](https://www.researchgate.net/publication/353392781_When_the_Timeline_Meets_the_Pipeline_A_Survey_on_Automated_Cyberbullying_Detection).  <br>
The experiments were run in two different setups: one setup where the model tries to learn by using all the samples from the given labeled dataset, and another setup where the model is restricted to learn only from 10 samples from each category from the given labeled dataset. The first scenario is the classic approach where we have access to a full labeled dataset and to a lot more unlabeled data. The second scenario is a more uncommon one where we have access to only a few samples, but again a lot more unlabeled data. <br>
The architecture used is a simple one based on a pretrained BERT and a Classification Head built on top of it. The BERT model will have the weights unfrozen and will act as a feature extractor. The Classification Head is made out of 2 sequential Linear Layers. <br>
For the moment, the augmentations supported are:
- Backtranslation English-to-German,
- Backtranslation English-to-Russian,
- EDA_01 -> only 10% of the words from a sample are affected by the EDA operations,
- EDA_02 -> only 20% of the words from a sample are affected by the EDA operations,
- EDA_SR -> applies only the synonym replacement operation on 10% of the words from a sample. 

## Disclaimer
The data used in this project includes instances of sexism and hate speech. Therefore, reader discretion is strongly advised. The contributors to this project strongly oppose discrimination based on gender, religion, race or any other kind. One of the goals of this project is to raise awareness about cyberbullying's negative effects.

## Project Structure
```
.
│   .gitignore
│   README.md
|   setup.py
│
├───.vscode
│       settings.json
│
└───fixtext                      <- All necessary code for Fix Text. 
    │   test.py                      <- Test function for evaluating the models.
    │   train.py                     <- Train function for Fix Text.
    │   __init__.py
    │
    ├───augmentations                <- Augmentations. 
    │       backtranslate.py             <- Computes backtranslation.
    │       eda.py                       <- Defines operations for EDA Augmentation.
    │       precompute_eda.py            <- Precomputes EDA Augmentation for a given dataset.    
    │       stop_words.py                <- Contains a list of stop words.
    │       utils.py                     <- Utils function for computing EDA.
    │       __init__.py
    │
    ├───data                         <- Dataset.
    │       get_datasets.py              <- Functions for building JSONL Datasets and Data Loaders.
    │       text_dataset.py              <- Defines JsonlDataset Class.
    │       vocab.py                     <- Defines Vocabulary Class.
    │       __init__.py
    │
    ├───models                       <- Models.
    │       bert.py                      <- Defines ClassificationBert Class.
    │       trainer_helpers.py           <- Functions for extracting the optimizer and scheduler.
    │       __init__.py
    │
    └───utils                        <- Utils Functions.
            argument_parser.py           <- Argument Parser for the training script.
            average_meter.py             <- AverageMeter Class to save statistics regarding metrics.
            metrics.py                   <- Defines metrics of interest.
            utils.py                     <- Other utils functions.
            __init__.py
```

## Getting Started
### Setup
In order to setup the working space:
- git clone this repository:
```bash
git clone https://github.com/AndreiDumitrescu99/FixText.git
```
- make a virtual env where you will install all the needed packages, for example:
```bash
python3 -m venv dev_venv
source dev_venv/bin/activate
```
- intall the FixText module:
```bash
pip install -e .
```

### Dataset Preparation
You will need 4 different jsonl files: one for the labeled training set, one for the unlabeled training, one for the validation set and one for the testing set.
A sample from these files should have the following JSON format:
```json
{
    "text": "Some text",
    "textDE": "Backtranslated text with german.",
    "textRU": "Backtranslated text with russian.",
    "label": "Label of the sample.",
    "dataset": "From which dataset it belongs.",
    "split": "either: training, testing or validation",
    "eda_01": ["...", "...", ...],
    "eda_02": ["...", "...", ...],
    "eda_sr": ["...", "...", ...],
}
```
To obtain this format, you should start with a JSONL file where the samples have the following format:
```json
{
    "text": "Some text",
    "label": "Label of the sample.",
    "dataset": "From which dataset it belongs.",
    "split": "either: training, testing or validation",
}
```
Using the `backtranslate.py` script, you can generate the `textDE` and `textRU` properties.
Using the `precompute_eda.py` script, you can generate the `eda_01`, `eda_02` and `eda_sr` properties. <br>

### Training
Supposse you have 2 datasets: X, Y on which you want to run FixText, and you've run the steps presented above for dataset preparation.
The final folder structure where the datasets are stored should look like this:
```
.
│
├───dev
│       dataset_X.jsonl
|       dataset_Y.jsonl
│
├───test
│       dataset_X.jsonl
|       dataset_Y.jsonl
│
├───train
│       dataset_X.jsonl
|       dataset_Y.jsonl
│
└───unlabeled
        unlabeled_dataset_X.jsonl
        unlabeled_dataset_Y.jsonl
```
Take a closer look at the arguments from the `argument_parser.py` and decide with what hyperparameters you want to run. <br>
Initially, if you want to run the FixText algorithm on the dataset X, simply run the `train.py` file with the desired arguments, or you can simply run with the default parameters by doing this:
```bash
python train.py --data_path path_to_datasets_folders --unlabeled_dataset unlabeled_dataset_X.jsonl --task dataset_X.jsonl --out path_to_output_folder
```

## Further Work
As further work, I would like:
- to add support for [Hydra](https://hydra.cc/docs/intro/)
- to better generalize the use case of Fix Text
- add unit tests for some functions
- add more augmentations