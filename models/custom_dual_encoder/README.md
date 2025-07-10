## Understanding the Configuration File for Training a Dual Encoder Model

This guide explains the structure and options in the configuration file used for training a dual encoder model.

### Example Config File

```ini
[General]
outdir = ../../models/custom_dual_encoder
model_name = custom_dual_encoder
suffix = aida-training

[Data]
corpus_name = aida
label_type = nel
document_level = true
verbalizations_path = structured_verbalizations_zelda_labels.json
labels_path = None
use_corpus_labels = True
replace_labels = True
replace_verbalizations = True
verbalization_strategy = title+description
max_verbalization_length_soft = 50

[Model]
greedy = false
insert_in_context = 0
insert_which_labels = gold_pred
label_embedding_batch_size = 128
transformer = bert-base-uncased
pooling = first_last
context_size = 0
negative_sampling_strategy = hard
negative_sampling_factor = 1
loss = cross_entropy
similarity_metric = euclidean
constant_updating = True

[Training Parameters]
epochs = 10
batch_size = 32
batch_size_eval = 32
learning_rate = 5e-6
seed = 123
```

### Sections

#### [General]

| Key          | Description                                                                |
| ------------ |----------------------------------------------------------------------------|
| `outdir`     | Output directory for saving the model folder with checkpoints and logs.    |
| `model_name` | Name to use for the model folder.                     |
| `suffix`     | Optional string to distinguish model runs. Gets appended to the model name. |

#### [Data]

| Key                             | Description                                                                                                                                             |
| ------------------------------- |---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `corpus_name`                   | The name or path of the training corpus. Use `'aida'`, `'zelda'`, a filename in the `datasets/` folder, or a full path.                                 |
| `label_type`                    | The column name used for labels (e.g., `nel`, `mentions`).                                                                                              |
| `document_level`                | Set to `true` if training should use full documents instead of sentence-level data.                                                                     |
| `verbalizations_path`           | JSON file containing label verbalizations in the `data/verbalizations`folder, default is `structured_verbalizations_zelda_labels.json`.                 |
| `labels_path`                   | Optional file in the `data/verbalizations`folder with a list of labels (one label per line, see e.g. `demo_labels.txt`) to restrict training/prediction. |
| `use_corpus_labels`             | Whether to instead use all labels that are present in the corpus as label set.                                                                          |
| `replace_labels`                | If `True`, replaces VerbalizED label set with those from `labels_path` or the corpus labels. if `False, they get _added_ istead.                        |
| `replace_verbalizations`        | If `True`, replaces or overrides VerbalizED verbalizations with those from the file. If `False, they get _added_ instead.                               |
| `verbalization_strategy`        | Defines how labels are verbalized (e.g., `title`, `title+description`, `title+description+category`).                                                   |
| `max_verbalization_length_soft` | Maximum length (in characters) after wich the verbalization is cut (_soft_ meaning, not inside a word).                                                   |

#### [Model]

| Key                          | Description                                                                                                                                                   |
| ---------------------------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `greedy`                     | If `True`, uses the iterative VerbalizED variant, default `False` uses the base variant.                                                                      |
| `insert_in_context`          | Wheather the insertions should affect also the neighbouring sentences (0 = no, 1 = 1 sentence before and after, 2 = sentences before ad after). Default is 0. |
| `insert_which_labels`        | Which labels to insert during training: `gold`, `pred`, or `gold_pred` (starting with gold and switching to pred lader).                                      |
| `label_embedding_batch_size` | Batch size for embedding label descriptions.                                                                                                                  |
| `transformer`                | The HuggingFace transformer model to use (e.g., `bert-base-uncased`).                                                                                         |
| `pooling`                    | How to pool token embeddings (`first_last`, `first`, `last` or `mean`).                                                                                       |
| `context_size`               | Max number of tokens from surrounding sentences to include.                                                                                                   |
| `negative_sampling_strategy` | Strategy for sampling negatives (`hard`, `random`, `shift` (=in-batch).                                                                                       |
| `negative_sampling_factor`   | How many negative examples per positive, either int or `dyn`.                                                                                                 |
| `loss`                       | Loss function to use (e.g., `cross_entropy`, `triplet).                                                                                                       |
| `similarity_metric`          | Distance metric to use (`cosine`, `euclidean`, mm`).                                                                                                          |
| `constant_updating`          | If `true`, updates label embeddings frequently and dynamically (if label is seen).                                                                            |

#### [Training Parameters]

| Key               | Description                        |
| ----------------- | ---------------------------------- |
| `epochs`          | Number of training epochs.         |
| `batch_size`      | Training batch size.               |
| `batch_size_eval` | Batch size used during evaluation. |
| `learning_rate`   | Optimizer learning rate.           |
| `seed`            | Random seed for reproducibility.   |

### Notes

- Any key not specified will fall back to its default (if defined in the code).
- JSON and text files (like verbalizations or labels) must be properly formatted.