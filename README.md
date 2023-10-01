# Fix Text
Fix Text extends the idea behind of [Fix Match](https://arxiv.org/ftp/arxiv/papers/2001/2001.07685.pdf) to text based inputs. The proposed method also builds upon the work done in the ["Multimodal Semi-supervised Learning for Disaster Tweet Classification”](https://aclanthology.org/2022.coling-1.239.pdf) research paper by extracting the main ideas behind the text augmentations. The algorithm for Fix Text is really similar to the Fix Match one, the only main difference being the augmentations we apply on the input text. This code was used to produce the results presented in this [paper](https://drive.google.com/file/d/1UxUAsKJvt7pgwdc7eqLMTxTx6SgjR2Or/view?usp=sharing).

## Our Approach - Cyberbullying Detection Task
Our goal is to to compare how models trained in a semi supervised manner using the Fix Text algorithm compare with the models trained in
a classic supervised way on a concrete task. <br>
Fix Text was tested on the Cyberbullying Detection task on 4 different datasets, in 2 separate scenarios. Each dataset has a different data distribution from the others, and they try to cover distinct social media contexts: Tweets, YouTube Comments and Kaggle Comments. When training the model in a semi supervised way on one of the 4 datasets will act as a labeled dataset, the rest 3 of them will act as an unlabeled dataset. The datasets used are: the [Cyberbullying Tweets dataset](https://people.cs.vt.edu/ctlu/Publication/2020/IEEE-BD-SOSNet-Wang.pdf), [Kaggle Comments dataset ](https://www.researchgate.net/publication/353392781_When_the_Timeline_Meets_the_Pipeline_A_Survey_on_Automated_Cyberbullying_Detection), [YouTube Comments dataset](https://www.researchgate.net/publication/353392781_When_the_Timeline_Meets_the_Pipeline_A_Survey_on_Automated_Cyberbullying_Detection) and another [Tweets Dataset](https://www.researchgate.net/publication/353392781_When_the_Timeline_Meets_the_Pipeline_A_Survey_on_Automated_Cyberbullying_Detection).  <br>
The experiments were run in two different setups: one setup where the model tries to learn by using all the samples from the given labeled dataset, and another setup where the model is restricted to learn only from 10 samples from each category from the given labeled dataset. The first scenario is the classic approach where we have access to a full labeled dataset and to a lot more unlabeled data. The second scenario is a more uncommon one where we have access to only a few samples, but again a lot more unlabeled data. <br>
The architecture used is a simple one based on a pretrained BERT and a Classification Head built on top of it. The BERT model will have the weights unfrozen and will act as a feature extractor. The Classification Head is made out of 2 sequential Linear Layers.

## Disclaimer
The data used in this project includes instances of sexism and hate speech. Therefore, reader discretion is strongly advised. The contributors to this project strongly oppose discrimination based on gender, religion, race or any other kind. One of the goals of this project is to raise awareness about cyberbullying's negative effects.

## Project Structure
```
.
│   .gitignore
│   README.md
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
