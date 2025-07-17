import os
from flair.models import DualEncoderEntityDisambiguation, GreedyDualEncoderEntityDisambiguation
from pathlib import Path
import flair
from configparser import ConfigParser
import argparse
import json

from verbalizations_loader import load_labels_from_file, read_in_verbalizations_and_do_cutoff, normalize_verbalization_strategy
from dataset_loader import infer_and_load_dataset

VERBALIZATIONS_PATH = Path("../data/verbalizations")

DATASETS_PATH = Path("../data/datasets/")

MODELS_PATH = Path("../models/")


parser = argparse.ArgumentParser(description="Dataset loader and verbalization handling.")
parser.add_argument(
    '--model',
    type=str,
    default="base",
    help='Variant of the model to use, must be either one of ["base", "iterative"] for loading our VerbaliZED models or the name of your custom model (e.g. "custom_dual_encoder/my_model").'
)

parser.add_argument(
    '--name',
    type=str,
    default="default",
    help='Optional name for the evaluation subdirectory, choose one to prevent overwriting previous evaluation results (e.g., "adding-corpus-labels"). If not given, the default name "default" will be used.'
)

parser.add_argument(
    '--datasets',
    nargs='+',
    type=str,
    help='Space-separated list of dataset names or paths (e.g., --dataset aida custom1.tsv /path/to/custom2.tsv)'
)

parser.add_argument(
    '--mention-detection',
    type=bool,
    default=False,
    help='Set to True to first run a mention detection model on the dataset(s) before running entity disambiguation. This is necessary if the dataset does not contain mention annotations.'
)

parser.add_argument(
    '--column-format',
    type=json.loads,
    required=False,
    default='{"0": "text", "1": "nel"}',
    help='JSON string defining column format (applies to all datasets), e.g., \'{"0": "text", "1": "nel"}\''
)

parser.add_argument(
    '--label-type',
    type=str,
    required=False,
    default = "nel",
    help='Label type / name of column to use (e.g., nel, ner, to_predict) (important: applies to all given datasets!)'
)

parser.add_argument(
    '--label-list',
    type=str,
    default=None,
    help='Path to a .txt file with one label per line (optional). If none give, the labels that the model was trained on will be used.'
)

parser.add_argument(
    '--replace-labels',
    type=bool,
    default=True,
    help='Set to True to replace labels with label-list, so *only* allowing these labels. If False, label-list are *added* to the existing ones.'
)

parser.add_argument(
    '--verbalization-dict',
    type=str,
    default=None,
    help='Path to a .json file with label: verbalization mappings (optional)'
)

parser.add_argument(
    '--verbalization-strategy',
    type=str,
    default="title+descriptions+categories",
    help='Verbalization strategy to use, e.g., "title+descriptions+categories". Options: "title", "descriptions", "categories" (or "classes"). Use "+" to combine them.'
)

parser.add_argument(
    '--verbalization-length',
    type=int,
    default=50,
    help='Maximum length of verbalizations in characters. If longer, they will be cut off. Default is 50 characters.'
)

parser.add_argument(
    '--replace-verbalizations',
    type=bool,
    default=True,
    help='Set to True to replace verbalizations, so *only* using the new verbalizations. If False, verbalizations are *added* to the existing ones.'
)

parser.add_argument(
    '--print-predictions-to-file',
    type=bool,
    default=True,
    help='Set to True to print predictions to a file in the model directory. If False, predictions are not saved as column file.'
)

parser.add_argument(
    '--top-k',
    type=int,
    default=1,
    help='Number of top-k predictions to save. Default is 1, meaning only the best prediction will be saved.'
)

args = parser.parse_args()

if args.column_format:
    args.column_format = {int(k): v for k, v in args.column_format.items()}

if not args.datasets:
    raise ValueError("You must provide at least one dataset using the --datasets argument.")
print("Datasets provided:", args.datasets)

#args.datasets = ["my_custom_NER_set"] # TODO
#args.column_format = {0: "text", 1: "nel"} # TODO
#args.model = "custom_dual_encoder/trained_on_custom_dataset/corpus-labels"
#args.model = "base"

flair.device = "cuda:0"

# DEFINE MODEL PATHS
if args.model == "base":
    model_identifier = "VerbalizED_base_ZELDA"
elif args.model == "iterative":
    model_identifier = "VerbalizED_iterative_ZELDA"
else:
    #raise ValueError(f"Unknown model '{args.model}'. Must be one of ['base', 'iterative'].")
    print("Using custom model identifier:", args.model)
    model_identifier = args.model

model_path = MODELS_PATH / model_identifier / "final-model.pt"
print("Using model path:", model_path)
config_path = MODELS_PATH / model_identifier / "config.ini"
print("Using configuration file:", config_path)

# DEFINE EVAL PATH
eval_path =  MODELS_PATH / model_identifier / "evaluation" / args.name
print("Using evaluation path:", eval_path)

# READ SPECIFICATIONS FROM CONFIGURATION FILE
parser = ConfigParser()
# if config file not exist, raise ValueError
if not os.path.exists(config_path):
    raise ValueError(f"Configuration file {config_path} does not exist. Please make sure that the correct path is given.")
parser.read(config_path)

config_args = {
    "outdir": parser.get("General", "outdir"),
    "corpus_name": parser.get("Data", "corpus_name"),
    "document_level": parser.getboolean("Data", "document_level", fallback=False),
    "downsample_factor": parser.getfloat("Data", "downsample_factor", fallback=None),
    #"general_label_set": parser.getboolean("Data", "general_label_set"),
    "verbalizations_path": parser.get("Data", "verbalizations_path", fallback=None),
    "verbalization_strategy": parser.get("Data", "verbalization_strategy"),
    "max_verbalization_length_soft": parser.getint("Data", "max_verbalization_length_soft", fallback=15),
    "greedy": parser.getboolean("Model", "greedy"),
    "insert_in_context": parser.getint("Model", "insert_in_context", fallback=0),
    "insert_which_labels": parser.get("Model", "insert_which_labels", fallback="gold"),
    "label_embedding_batch_size": parser.getint("Model", "label_embedding_batch_size", fallback=None),
    "label_sample_size": parser.getint("Model", "label_sample_size", fallback=None),
    "transformer": parser.get("Model", "transformer"),
    "pooling": parser.get("Model", "pooling"),
    "context_size": parser.getint("Model", "context_size"),
    "negative_sampling_strategy": parser.get("Model", "negative_sampling_strategy"),
    "negative_sampling_factor": parser.get("Model", "negative_sampling_factor"),
    "loss": parser.get("Model", "loss"),
    "similarity_metric": parser.get("Model", "similarity_metric", fallback="euclidean"),
    "constant_updating": parser.getboolean("Model", "constant_updating", fallback=False),
    "device": parser.get("Training Parameters", "device", fallback=None),
    "epochs": parser.getint("Training Parameters", "epochs"),
    "batch_size": parser.getint("Training Parameters", "batch_size"),
    "batch_size_eval": parser.getint("Training Parameters", "batch_size_eval"),
    "learning_rate": parser.getfloat("Training Parameters", "learning_rate"),
    "continue_from_checkpoint": parser.getboolean(
        "Training Parameters", "continue_from_checkpoint", fallback=False
    ),
    "checkpoint_name": parser.get("Training Parameters", "checkpoint_name", fallback=None),
    "seed": parser.getint("Training Parameters", "seed", fallback=123)
}

print("Loading from state dict:")
print(model_path)

# SET THE MODEL SPECIFICATIONS
if config_args["greedy"] == True:
    print("Using iterative model")
    model = GreedyDualEncoderEntityDisambiguation.load(model_path=model_path)
else:
    print("Using base (i.e. not iterative) model")
    model = DualEncoderEntityDisambiguation.load(model_path=model_path)

# LOADING DATASETS
evaluate_on = {} # will be a dict with dataset names as keys, each with datasets and label types

for user_input in args.datasets:
    if args.mention_detection:
        args.column_format = {0: "text"}
    dataset_info = infer_and_load_dataset(user_input,
                                          document_level=config_args["document_level"],
                                          dataset_path= DATASETS_PATH,
                                          column_format= args.column_format,
                                          label_type=args.label_type,
                                          only_test=True)

    evaluate_on.update({k: {"data": v["data"],
                            "label_type": v["label_type"]}
                                  for k,v in dataset_info.items()})

print("Evaluating on the following datasets:")
print(evaluate_on.keys())

model.token_encoder.allow_long_sentences = True
model.similarity_metric.metric_to_use = config_args["similarity_metric"]
model.embedding_pooling = config_args["pooling"]

print("Using the following specifications:")
print("Similarity metric:", model.similarity_metric.metric_to_use)
print("Loss:", model.loss_function)
print("Embedding Pooling:", model.embedding_pooling)
print("Greedy:", config_args["greedy"])
print("Document:", config_args["document_level"])
print("Model was trained on the Verbalization Strategy:", config_args["verbalization_strategy"])
print("Max Len:", config_args["max_verbalization_length_soft"])

if args.label_list is not None:
    print("Loading additional labels from:", VERBALIZATIONS_PATH / args.label_list)
    new_labels = load_labels_from_file(VERBALIZATIONS_PATH / args.label_list)

    # Replace or Add the new labels and verbalizations?
    if args.replace_labels:
        print("Replacing the known labels with the new ones.")
        model.known_labels = new_labels
    else:
        print("Adding the new labels to the known labels.")
        model.known_labels.extend(new_labels)
else:
    print(f"No additional labels provided, using the model's known labels. ({len(model.known_labels)} known labels)")

if args.verbalization_dict is None:
    print("No verbalization dictionary provided, using the verbalization dict the model was trained on.")

    if normalize_verbalization_strategy(args.verbalization_strategy) != normalize_verbalization_strategy(config_args["verbalization_strategy"]):
        print("Warning: You provided a verbalization strategy that is different from the one used to train the model. This might lead to unexpected results.")
        print("Model was trained on the verbalization strategy:", config_args["verbalization_strategy"])
        print("Now using the verbalization strategy:", args.verbalization_strategy)
        print("On the verbalizations provided here:", config_args["verbalizations_path"])
        new_verbalization_dict = read_in_verbalizations_and_do_cutoff(VERBALIZATIONS_PATH / config_args["verbalizations_path"],
                                                                      args.verbalization_strategy,
                                                                      config_args["max_verbalization_length_soft"])
    else:
        print("Using the verbalization dictionary the model was trained on.")
        new_verbalization_dict = None

else:
    print("Loading verbalization dictionary from:", VERBALIZATIONS_PATH / args.verbalization_dict)
    new_verbalization_dict = read_in_verbalizations_and_do_cutoff(VERBALIZATIONS_PATH / args.verbalization_dict,
                                                                  args.verbalization_strategy,
                                                                  config_args["max_verbalization_length_soft"])

if new_verbalization_dict:
    if args.replace_verbalizations:
        print("Replacing the known verbalizations with the new ones.")
        model.label_map = new_verbalization_dict
    else:
        print("Adding the new verbalizations to the known verbalizations, favouring the new ones if the key already exists.")
        model.label_map.update(new_verbalization_dict)

if args.label_list is not None and args.verbalization_dict is None:
    print("Warning: You provided a label list but no verbalization dictionary. The model will use the VerbalizED label dictionary. Labels not in the verbalization dictionary will be verbalized with only using the title.")

if args.label_list is None and args.verbalization_dict is not None:
    print("Warning: You provided a verbalization dictionary but no label list. The model will use the VerbalizED label list. Labels not in the label list will not be used.")

# PREDICT ON THE DATA


def print_to_column_file(sentences, target_path, label_type, save_top_k = None):
    with open(target_path, "w", encoding="utf-8") as out:
        for nr, sentence in enumerate(sentences):
            # convert to token level BIO tagging
            span_label_types = [label_type, "predicted"]
            if save_top_k:
                for k in range(1, save_top_k + 1):
                    span_label_types.append(f"top_{k}")

            for span_label_type in span_label_types:
                spans = sentence.get_spans(span_label_type)
                for s in spans:
                    label = s.get_label(span_label_type).value
                    score = round(s.get_label(span_label_type).score, 3)
                    if label == "O":
                        for i in range(s[0].idx - 1, s[-1].idx):
                            sentence[i].set_label(f"{span_label_type}_BIO", "O")
                    else:
                        sentence[s[0].idx - 1].set_label(f"{span_label_type}_BIO", "B-" + label)
                        sentence[s[0].idx - 1].set_label(f"{span_label_type}_score", score)
                        for i in range(s[0].idx, s[-1].idx):
                            sentence[i].set_label(f"{span_label_type}_BIO", "I-" + label)
                            sentence[i].set_label(f"{span_label_type}_score", score)

            # print everything per token/line
            for token in sentence:
                token_text = token.text
                if not save_top_k:
                    out.write(
                        f"{token_text}\t"
                        f"{token.get_label(f'{label_type}_BIO').value}\t"
                        f"{token.get_label('predicted_BIO').value}, {token.get_label('predicted_score').value}\n"
                    )
                else:
                    out.write(
                        f"{token_text}\t"
                        f"{token.get_label(f'{label_type}_BIO').value}"
                    )
                    for k in range(1, save_top_k + 1):
                        out.write(f"\t{token.get_label(f'top_{k}_BIO').value}, {token.get_label(f'top_{k}_score').value}")

                    out.write("\n")

            # print newline at end of each sentence, not after last sentence
            if nr < len(sentences) - 1:
                out.write("\n")


for name, entry in evaluate_on.items():
    print("###############################################################################")
    print(f"Evaluating on {name} ...")

    sentences = entry["data"]
    label_type = entry["label_type"]

    if args.mention_detection:
        print("First running Flair NER tagger for mention detection before EL evaluation.")
        from flair.nn import Classifier
        tagger = Classifier.load("ner")
        label_type = "ner"
        tagger.predict(sentences,
                       label_name=label_type)

        print("One example sentence after NER tagging, so with annotated mention spans:")
        print(sentences[0])

    model._label_type = label_type
    print("Using label type:", label_type)

    # make sure the directory exists
    current_eval_path = eval_path / name
    eval_path.mkdir(parents=True, exist_ok=True)

    # evaluate the model (and save the predictions to disk)
    results = model.evaluate([s for s in sentences],
                             out_path=eval_path / f"{name}_predictions.txt",
                             gold_label_type=label_type,
                             return_loss=True,
                             save_top_k=args.top_k,
                            )
    print("--> Main Score:", results.main_score)
    #print(results)

    # save results
    print(f"Saving more detailed results to {eval_path / f'{name}_results.txt'}")
    with open(eval_path / f"{name}_results.txt", "w") as f:
        f.write(str(results))

    # print predictions to column file
    if args.print_predictions_to_file:
        print(f"Printing predictions to {eval_path / f'predictions_column_format_{name}.tsv'}")
        print_to_column_file(sentences, eval_path / f"predictions_column_format_{name}.tsv", label_type=label_type, save_top_k=args.top_k)

    # saving the scores per dataset
    evaluate_on[name]["Scores"] = results.scores

    print_examples = False
    if print_examples:
        # print some examples, note that we save the top-5 predictions as well
        print("Some examples:")
        for i in range(3):
            print("---")
            print(sentences[i].text)
            for sp in sentences[i].get_spans(label_type):
                print(sp.text)
                print("Gold:", sp.get_label(label_type).value)
                print("Pred:", sp.get_label("predicted").value)
                top_5 = [sp.get_label("top_0").value, sp.get_label("top_1").value, sp.get_label("top_2").value,
                         sp.get_label("top_3").value, sp.get_label("top_4").value]
                print(f"  Top 5 were: {top_5}")
                print("-")

print("Done with evaluation. Results per dataset are saved in the evaluation path:", eval_path)

# Path to save the TSV file
output_tsv_path = eval_path / "results_overview.tsv"
print("Writing results overview to:", output_tsv_path)

accuracies = []

with open(output_tsv_path, "w") as f:
    # Write header
    header = "Dataset\tAccuracy\n"
    f.write(header)
    print(header.strip())  # Print without trailing newline

    # Write data rows
    for name in evaluate_on:
        accuracy = evaluate_on[name]['Scores'].get('accuracy', 'N/A')

        if isinstance(accuracy, float):
            accuracies.append(accuracy)

        row = f"{name}\t{accuracy}\n"
        f.write(row)
        print(row.strip())  # Print without trailing newline

    # Calculate mean
    mean_accuracy = round(sum(accuracies) / len(accuracies), 4) if accuracies else "N/A"

    # Write mean row
    mean_row = f"mean\t{mean_accuracy}\n"
    f.write(mean_row)
    print(mean_row.strip())