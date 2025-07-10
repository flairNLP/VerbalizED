# VerbalizED

This repository contains the code and resources supporting our ACL 2025 paper **"[Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation](https://arxiv.org/pdf/2505.11683)"**.

## Overview

Entity disambiguation (ED) is a crucial task in natural language understanding, aiming to correctly link ambiguous mentions in text to their corresponding entities in a knowledge base. A popular architecture are Dual Encoder (aka Bi-Encoders). In our work, we evaluated various design decisions within dual encoder architectures for ED, providing insights for improving model performance. Our resulting system VerbalizED combines structured label verbalizations, hard negative mining and efficient label embeddig updates.

This repository provides the code for creating verbalizations for new entity labels, training custom Dual Encoder models, and making predictions with pre-trained VerbalizED models. It also includes the datasets and verbalizations used in our experiments.

## Repository Structure

```
├── data/                             # Datasets, labels and verbalizations
│   ├── verbalizations/               # Labels and verbalizations
│   │   ├── demo_labels.txt
│   │   ├── demo_labels_verbalizations.txt
│   │   ├── zelda_labels_verbalizations.txt
│   │   └── zelda_labels.txt
│   └── datasets/                     # Evaluation and training datasets
│       ├── zelda_test/                          # the ZELDA test sets
│       ├── my_custom_dataset/                   # Custom dataset for training, has to include a train.tsv
│       └── my_custom_test_set.tsv               # Custom test set for evaluation

├── scripts/                          # Scripts for training and prediction
│   ├── label_verbalizer.py                      # Create verbalizations for new label sets
│   ├── verbalizations_loader.py                 # Helper: loads verbalization files
│   ├── dataset_loader.py                        # Helper: loads datasets in expected format
│   ├── train_custom_dual_encoder.py             # Train a custom Dual Encoder model
│   └── load_and_predict_with_trained_model.py   # Load a VerbalizED or custom model and make predictions

├── models/                           # Saved model checkpoints
│   ├── VerbalizED_base_ZELDA/                   # Base VerbalizED model trained on ZELDA
│   ├── VerbalizED_biterative_ZELDA/             # Iterative VerbalizED model trained on ZELDA
│   ├── custom_dual_encoder/                     # Custom model outputs (config, checkpoints, predictions)
│       └── config_aida.ini                      # Example config file to control training behavior
│   └── [more models]                            # Your saved models will be added here

├── README.md                         # Project overview and usage instructions
├── .gitignore                        # Git ignore file
└── requirements.txt                  # Python package dependencies

```


## Getting Started

### General

- Our code for our Dual Encoder model is included in [**`flair`**](https://github.com/flairNLP/flair), currently in the branch [dual_encoder_similarity_loss](https://github.com/flairNLP/flair/blob/dual_encoder_similarity_loss/flair/models/dual_encoder_ED.py), we will merge it to main soon and update this tutorial accordingly, stay tuned!
- The branch is included in the requirements.txt file, so you can install it via `pip install -r requirements.txt`, see below.

### Installation

* Crete a conda environment and install the required packages:

``` bash
conda create -n verbalized python=3.8
conda activate verbalized
pip install -r requirements.txt
```

## Usage
* There are 3 key ways to use this repository:
  * **Load and predict with VerbalizED**: Use the pre-trained VerbalizED models to make predictions on common test sets or your own new data.
  * **Train a custom Dual Encoder model**: Use the provided script to train your own model on a dataset of your choice.
  * **Gather verbalizations for new labels**: Use the provided script to create verbalizations for new entity labels.

### Verbalize a list of new labels:

* If you have a custom list of labels, you can create verbalizations for them using the `label_verbalizer.py` script. This script will generate verbalizations based on the provided label file and save them in a JSON format.
* Try it out with the demo labels (`demo_labels_list.txt`):

``` bash
python label_verbalizer.py --label_file demo_labels_list.txt --output demo_labels_verbalizations.json
```

* The output will be a JSON file containing the verbalization entries for each label (`demo_labels_verbalizations.json`). These can be used for training or predicting with the Dual Encoder model. It looks like:
``` bash
{
  "Albert_Einstein": {
    "wikidata_description": "German-born theoretical physicist (1879\u20131955)",
    "wikidata_id": "Q937",
    "wikidata_properties": [
      ["instance of",
       "human"
      ],
      ["occupation",
       "physicist"
      ]
    ],
    "wikidata_url": "https://www.wikidata.org/wiki/Q937",
    "wikipedia_description": "German-born theoretical physicist (1879\u20131955)",
    "wikipedia_paragraph": "Albert Einstein (14 March 1879 \u2013 18 April 1955) was a German-born theoretical physicist who is best known for developing the theory of relativity. Einstein also made important contributions to quantum mechanics. His mass\u2013energy equivalence formula E = mc2, which arises from special relativity, has been called \"the world's most famous equation\". He received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect.\nBorn in the German Empire, Einstein moved to Switzerland in 1895, forsaking his German citizenship (as a subject of the Kingdom of W\u00fcrttemberg) the following year. In 1897, at the age of seventeen, he enrolled in the mathematics and physics teaching diploma program at the Swiss federal polytechnic school in Zurich, graduating in 1900. He acquired Swiss citizenship a year later, which he kept for the rest of his life, and afterwards secured a permanent position at the Swiss Patent Office in Bern. In 1905, he submitted a successful PhD dissertation to the University of Zurich. In 1914, he moved to Berlin to join the Prussian Academy of Sciences and the Humboldt University of Berlin, becoming director of the Kaiser Wilhelm Institute for Physics in 1917; he also became a German citizen again, this time as a subject of the Kingdom of Prussia. In 1933, while Einstein was visiting the United States, Adolf Hitler came to power in Germany. Horrified by the Nazi persecution of his fellow Jews, he decided to remain in the US, and was granted American citizenship in 1940. On the eve of World War II, he endorsed a letter to President Franklin D. Roosevelt alerting him to the potential German nuclear weapons program and recommending that the US begin similar research.\nIn 1905, sometimes described as his annus mirabilis (miracle year), he published four groundbreaking papers. In them, he outlined a theory of the photoelectric effect, explained Brownian motion, introduced his special theory of relativity, and demonstrated that if the special theory is correct, mass and energy are equivalent to each other. In 1915, he proposed a general theory of relativity that extended his system of mechanics to incorporate gravitation. A cosmological paper that he published the following year laid out the implications of general relativity for the modeling of the structure and evolution of the universe as a whole. In 1917, Einstein wrote a paper which introduced the concepts of spontaneous emission and stimulated emission, the latter of which is the core mechanism behind the laser and maser, and which contained a trove of information that would be beneficial to developments in physics later on, such as quantum electrodynamics and quantum optics.\nIn the middle part of his career, Einstein made important contributions to statistical mechanics and quantum theory. Especially notable was his work on the quantum physics of radiation, in which light consists of particles, subsequently called photons. With physicist Satyendra Nath Bose, he laid the groundwork for Bose\u2013Einstein statistics. For much of the last phase of his academic life, Einstein worked on two endeavors that ultimately proved unsuccessful. First, he advocated against quantum theory's introduction of fundamental randomness into science's picture of the world, objecting that God does not play dice. Second, he attempted to devise a unified field theory by generalizing his geometric theory of gravitation to include electromagnetism. As a result, he became increasingly isolated from mainstream modern physics."
  },
  "All_England_Club": {
    "wikidata_description": "English tennis club",
    "wikidata_id": "Q815369",
    "wikidata_properties": [
      ["country",
       "United Kingdom"
      ],
      ["instance of",
       "sports club"
      ],
      ["instance of",
       "gentlemen's club"
      ]
    ],
    "wikidata_url": "https://www.wikidata.org/wiki/Q815369",
    "wikipedia_description": "Private members' club in Wimbledon, England",
    "wikipedia_paragraph": "The All England Lawn Tennis and Croquet Club, also known as the All England Club, is a private members' club based at Church Road in the Wimbledon area of London, England. It is best known as the venue for the Wimbledon Championships, the only Grand Slam tennis event still held on grass. Initially an amateur event that occupied club members and their friends for a few days each summer, the championships have become far more prominent than the club itself.\nThe club has 375 full members, about 100 temporary playing members, and a number of honorary members. To become a full or temporary member, an applicant must obtain letters of support from four existing full members, two of whom must have known the applicant for at least three years. The name is then added to the candidates' list. Honorary members are elected from time to time by the club's committee. Membership carries with it the right to purchase two tickets for each day of the Wimbledon Championships. In addition to this, all champions are invited to become members.\nCatherine, Princess of Wales, has been the patron of the club since 2016, and took over in 2021 from HRH The Duke of Kent when he stepped down as president of the club, among a number of royal patronages."
  },
  "British_Museum": {
    "wikidata_description": "national museum in London, United Kingdom",
    "wikidata_id": "Q6373",
    "wikidata_properties": [
      ["country",
       "United Kingdom"
      ],
      ["instance of",
       "museum"
      ],
      ["instance of",
       "art museum"
      ],
      ["instance of",
       "non-departmental public body"
      ],
      ["instance of",
       "national museum"
      ]
    ],
    "wikidata_url": "https://www.wikidata.org/wiki/Q6373",
    "wikipedia_description": "National museum in London, England",
    "wikipedia_paragraph": "The British Museum is a public museum dedicated to human history, art and culture located in the Bloomsbury area of London. Its permanent collection of eight million works is the largest in the world. It documents the story of human culture from its beginnings to the present. Established in 1753, the British Museum was the first public national museum. In 2023, the museum received 5,820,860 visitors, 42% more than the previous year. At least one group rated it the most popular attraction in the United Kingdom.\nAt its beginning, the museum was largely based on the collections of the Anglo-Irish physician and scientist Sir Hans Sloane. It opened to the public in 1759, in Montagu House, on the site of the current building. The museum's expansion over the following 250 years was largely a result of British colonisation and resulted in the creation of several branch institutions, or independent spin-offs, the first being the Natural History Museum in 1881. Some of its best-known acquisitions, such as the Greek Elgin Marbles and the Egyptian Rosetta Stone, are subject to long-term disputes and repatriation claims.\nIn 1973, the British Library Act 1972 detached the library department from the British Museum, but it continued to host the now separated British Library in the same Reading Room and building as the museum until 1997. The museum is a non-departmental public body sponsored by the Department for Culture, Media and Sport. Like all UK national museums, it charges no admission fee except for loan exhibitions."
  },
  # more entities...
}
```



### Load a VerbalizED model and predict:
* Using the base (i.e. not iterative) VerbalizED model trained on ZELDA, and predicting for AIDA and Tweeki datasets:
* [TODO] talk about the corpus format
``` bash
python load_and_predict_with_trained_model.py \
  --model base \
  --datasets aida tweeki \
  --column-format '{"0": "text", "1": "nel"}' \
  --label-type 'nel'
```
* This will create three files per dataset, all placed in a `evaluation/eval-name` subdirectory of the model folder (e.g. `models/VerbalizED_base_Zelda/evaluation/eval-name`):
  1. The predictions per document. This will look like this, with the text, and each mention with the gold and predicted label, markings for correct (✓) and incorrect (❌) predictions, and the verbalization of the predicted one.
  ```
  SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . Nadim Ladki AL-AIN , United Arab Emirates 1996-12-06 
   - "JAPAN" / Japan_national_football_team --> Japan_national_football_team (✓) "Japan national football team; men's national association football team representing Japan" conf. -19.084 
   - "CHINA" / China_national_football_team --> China_national_football_team (✓) "China national football team; men's national association football team representing the People's" conf. -19.231 
   - "AL-AIN" / Al_Ain --> Al_Ain_FC (❌) "Al Ain FC; Emirati association football club; country: United Arab Emirates" conf. -19.788 
  ```
  2. The quantitative results (F score and Accuracy).
  ```
  Results:
  - F-score (micro) 0.8194
  - F-score (macro) 0.6901
  - Accuracy 0.8194
  [additional label-wise results]
  ```
  3. The predictions as a tsv file that allows for better readability and further processing, e.g. for evaluation with other tools. Each line contains the token, the gold label (or other label in the column specified by `label_type`, e.g. `mentions`) and the predicted label.
  ```
  SOCCER	O	O
  -	O	O
  JAPAN	B-Japan_national_football_team	B-Japan_national_football_team
  GET	O	O
  LUCKY	O	O
  WIN	O	O
  ,	O	O
  CHINA	B-China_national_football_team	B-China_national_football_team
  IN	O	O
  SURPRISE	O	O
  DEFEAT	O	O
  .	O	O
  Nadim	O	O
  Ladki	O	O
  AL-AIN	B-Al_Ain	B-Al_Ain_FC
  ```

* Also, an overview table with the main scores for all provided datasets will be created (`overview_results.tsv`), e.g.:
  ``` bash
  Dataset Accuracy
  test_reddit-comments    0.8603
  test_tweeki     0.7745
  test_shadowlinks-tail   0.9855
  test_cweb       0.6938
  test_aida-b     0.8207
  test_wned-wiki  0.9086
  test_reddit-posts       0.8834
  test_shadowlinks-shadow 0.6515
  test_shadowlinks-top    0.6604
  ```

* Now we want to evaluate on a different custom test set. We still use the default VerbalizED labels and verbalizations:
``` bash
python load_and_predict_with_trained_model.py \
  --model base \
  --datasets my_custom_test_set.tsv \
  --column-format '{"0": "text", "1": "nel"}' \
  --label-type 'nel'
```

* Now we want to use our custom labels and verbalizations that we created. We replace the labels and verbalizations, meaning we _only_ use them. We could also _add_ them to the existing VerbalizED ones (setting `replace-labels` and `replace-verbalizations` to `False`). We also give this evaluation a special name (`--name`) so it does not overwrite our previous runs.
``` bash
python load_and_predict_with_trained_model.py \
  --model base \
  --datasets my_custom_test_set.tsv \
  --column-format '{"0": "text", "1": "mentions"}' \
  --label-type 'mentions' \
  --label-list demo_labels_list.txt \
  --verbalization-dict demo_labels_verbalizations.json \
  --replace-labels True \
  --replace-verbalizations True
  --name custom-eval-corpus-labels
```

* The default verbalization strategy is to use the label names, the descriptions and the categories for the verbalizations. If you want to use a different strategy, you can specify it with the `--verbalization-strategy` argument (using `+` as connector, e.g. `title+description` or `title+categories`).
* But:
  * Keep in mind that not every label might have all elements, so only using `categories` might lead to problems.
  * Also, if you use the VerbalizED models, keep in mind that they were trained with one specific format. It is possible to switch it out for prediction, though this might lead to worse results than using the default strategy.
* For example, if you want to use titles and categories:
``` bash
python load_and_predict_with_trained_model.py \
  --model base \
  --datasets my_custom_test_set.tsv \
  --column-format '{"0": "text", "1": "mentions"}' \
  --label-type 'mentions' \
  --verbalization-strategy title+categories
  --name new-verbalization-strategy
```


#### Predicting on NER datasets

* Let's say we have a dataset, where _mentions_ are annotated (e.g. a NER dataset like `my_custom_NER_set.tsv`), but no entity disambiguation labels are present. We can still predict for it (though we need to ignore the resulting "scores"!):

``` bash
python load_and_predict_with_trained_model.py \
  --model base \
  --datasets my_custom_NER_set.tsv \
  --column-format '{"0": "text", "1": "entities"}' \
  --label-type 'entities'
```
* The resulting predictions will look like this:

  ``` bash
  NASA	B-ORG	B-NASA
  launched	O	O
  a	O	O
  new	O	O
  satellite	O	O
  from	O	O
  Cape	B-LOC	B-Cape_Canaveral_Space_Force_Station
  Canaveral	I-LOC	I-Cape_Canaveral_Space_Force_Station
  on	O	O
  Monday	O	O
  .	O	O
  ```
  
* Before using VerbalizED for Entity Disambiguation you need annotated mentions. Check out our many Flair NER taggers to get there!



### Train a custom Dual Encoder model
* You can also train your own Dual Encoder model.
* You can train on the AIDA or the ZELDA train sets, or on your own custom train dataset.
* As for training there are a lot of configurations, we opted for using a config file instead of command line arguments.
* Under `models/custom_dual_encoder`, you can find a config file that you can adapt to your needs.
* For example, to train on the AIDA train set (as the unmodified config uses the AIDA train set), you can run the following command:
``` bash
python train_custom_dual_encoder.py \
  --config config_aida.ini
```
* For training on your own custom dataset, modify the config (see `config_custom_dataset.ini` and the explanations [here](models/custom_dual_encoder/config-details.md)).
* The dataset argument needs to be a folder in the `datasets` folder, and it needs to have a `train.tsv` file in the correct format. See the `datasets/data/my_custom_dataset` folder as an example.
``` bash
python train_custom_dual_encoder.py \
  --config config_custom_dataset.ini
```

* More explanation about the content of the config file can be found [here](models/custom_dual_encoder/config-details.md).

### Loading your own custom dual encoder to predict with

* You can also load your trained model, similar to how you would load the VerbalizED models, but using the path to your model for the `--model` parameter. So say you want to evaluate your own model on AIDA as well as on a custom dataset:
``` bash
python load_and_predict_with_trained_model.py \
  --model custom_dual_encoder/my_model \
  --datasets aida my_custom_test_set.tsv
```


## Citation

If you use this code in your research, please cite our paper:

```
@article{rücker2025evaluatingdesigndecisionsdual,
        title={Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation}, 
        author={Susanna Rücker and Alan Akbik},
        year={2025},
        eprint={2505.11683},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2505.11683}, 
}
```

## Contact

For questions or feedback, please open an issue or contact:

- Susanna Rücker — susanna.ruecker@hu-berlin.de
