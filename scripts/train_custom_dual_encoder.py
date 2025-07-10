from flair.datasets import NEL_ENGLISH_AIDA, ZELDA, ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings, DocumentPoolEmbeddings
from flair.models import DualEncoderEntityDisambiguation, GreedyDualEncoderEntityDisambiguation
from flair.trainers import ModelTrainer
from pathlib import Path
import flair
from flair.distributed_utils import launch_distributed
import json
import torch
import argparse
from configparser import ConfigParser
from pathlib import Path
from typing import Union, List
import platform
import subprocess
import os

from verbalizations_loader import load_labels_from_file, read_in_verbalizations_and_do_cutoff
from dataset_loader import infer_and_load_dataset

flair.device = torch.device("cuda:0")
MULTI_GPU = False

VERBALIZATIONS_PATH = Path("../data/verbalizations")

DATASETS_PATH = Path("../data/datasets")

def cuda_devices_free_memory():
    result = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", shell=True)
    result = result.decode("utf-8")

    return [int(x) for x in result.strip().split("\n")]


def find_free_cuda_device_id(n: int = 1):
    free_mem = cuda_devices_free_memory()
    return sorted(range(len(free_mem)), key=lambda i: free_mem[i], reverse=True)[:n]

def train(
    outdir: str,
    model_name: str,
    corpus_name: str,
    label_type: str,
    suffix: str,
    document_level: bool,
    transformer: str,
    pooling: str,
    context_size: int,
    downsample_factor: Union[float, None],
    greedy: bool,
    verbalizations_path: str,
    labels_path: Union[str, None],
    use_corpus_labels: bool,
    replace_labels: bool,
    replace_verbalizations: bool,
    verbalization_strategy: str,
    label_embedding_batch_size: Union[int, None],
    label_sample_size: Union[int, None],
    negative_sampling_strategy: str,
    negative_sampling_factor: Union[float, str],
    device: str,
    epochs: int,
    batch_size: int,
    batch_size_eval: int,
    loss: str,
    similarity_metric: str,
    constant_updating: bool,
    learning_rate: float,
    continue_from_checkpoint: bool,
    checkpoint_name: str,
    seed: int,
    max_verbalization_length_soft: int,
    insert_in_context: Union[int, bool] = None,
    insert_which_labels: str = "gold",
):
    host = platform.node()
    print(f"Model runs on: {host}")

    if not MULTI_GPU:
        if not device:
            device_id = find_free_cuda_device_id()[0]
            device = f"cuda:{device_id}"

        flair.device = torch.device(device)
    print(f"Running on device: {flair.device}")

    flair.set_seed(seed)

    model_path = Path(outdir) / model_name / suffix

    print("Model path:", model_path)

    if verbalizations_path:
        print("Loading verbalizations from:", VERBALIZATIONS_PATH / verbalizations_path)
        custom_verbalizations_dict = read_in_verbalizations_and_do_cutoff(path=VERBALIZATIONS_PATH / verbalizations_path,
                                                                          verbalization_strategy=verbalization_strategy,
                                                                          max_verbalization_length_soft=max_verbalization_length_soft)
    else:
        print("No verbalizations path provided, using the VerbalizED verbalizations as fall back.")
        custom_verbalizations_dict = {}


    if not replace_verbalizations or len(custom_verbalizations_dict) == 0:
        # add the VerbalizED verbalizations
        verbalized_verbalizations_path = "zelda_labels_verbalizations.json"
        verbalized_verbalizations_dict = read_in_verbalizations_and_do_cutoff(path=VERBALIZATIONS_PATH / verbalized_verbalizations_path,
                                                                              verbalization_strategy=verbalization_strategy,
                                                                              max_verbalization_length_soft=max_verbalization_length_soft)

        # combine both, favouring the custom ones
        verbalizations_dict = verbalized_verbalizations_dict.copy()
        verbalizations_dict.update(custom_verbalizations_dict)

    else:
        verbalizations_dict = custom_verbalizations_dict

    print("Verbalizations dict size:", len(verbalizations_dict))

    # LOAD DATASET
    dataset_info = infer_and_load_dataset(corpus_name,
                                          dataset_path=DATASETS_PATH,
                                          label_type=label_type) # TODO

    name, values = next(iter(dataset_info.items()))
    print(f"Using dataset {name} for training")
    corpus = values["data"]

    if downsample_factor and downsample_factor < 1.0:
        corpus = corpus.downsample(downsample_factor, downsample_train=True, downsample_dev=False, downsample_test=False)

    print(corpus)

    # create label dictionary
    gold_label_dict = flair.data.Dictionary(
        add_unk=False
    )

    if labels_path and use_corpus_labels:
        raise ValueError("Cannot use custom labels and corpus labels at the same time. Please choose one of them.")
    if replace_labels:
        if not labels_path and not use_corpus_labels:
            raise ValueError("Cannot replace labels without providing a path to custom labels or use_corpus_labels. Please provide a labels_path.")
    if replace_verbalizations and not verbalizations_path:
        raise ValueError("Cannot replace verbalizations without providing a path to custom verbalizations. Please provide a verbalizations_path.")

    if labels_path:
        new_labels = load_labels_from_file(VERBALIZATIONS_PATH / labels_path)
        for l in new_labels:
            gold_label_dict.add_item(l)
        print(f"Loaded {len(new_labels)} custom labels from: {VERBALIZATIONS_PATH / labels_path}.")


    if use_corpus_labels:
        ADD_DEV_TEST = True  # this means, we collect the labels from the dev and test set as well, not only train
        corpus_label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=True, add_dev_test=ADD_DEV_TEST)
        for l in corpus_label_dict.get_items():
            gold_label_dict.add_item(l)
        print(f"Loaded {len(corpus_label_dict)} labels from the corpus.")

    if not replace_labels:
        print(f"Adding the VerbalizED (ZELDA) labels from {VERBALIZATIONS_PATH / 'zelda_labels.txt'}.")
        new_labels = load_labels_from_file(VERBALIZATIONS_PATH / 'zelda_labels.txt')
        for l in new_labels:
            gold_label_dict.add_item(l)
        print(f"Added {len(new_labels)} VerbalizED (ZELDA) labels.")

    if len(gold_label_dict) == 0:
        print("No custom or corpus labels provided, so loading VerbalizED (ZELDA) labels from:",
              VERBALIZATIONS_PATH / "zelda_labels.txt")
        new_labels = load_labels_from_file(VERBALIZATIONS_PATH / 'zelda_labels.txt')
        for l in new_labels:
            gold_label_dict.add_item(l)
        print(f"Added {len(new_labels)} VerbalizED (ZELDA) labels.")

    if "" in gold_label_dict.get_items():
        gold_label_dict.remove_item("")

    print(f"Gold label dict size: {len(gold_label_dict)}")

    if len(verbalizations_dict) < len(gold_label_dict):
        print(f"Warning: The verbalizations dict ({len(verbalizations_dict)}) is smaller than the gold label dict ({len(gold_label_dict)}).")
        print("Some labels will not have verbalizations, for those, only the title will be used.")
        print("If you don't want this, you can either create and load more verbalizations, or (if not already done) change replace_labels to False (to add the VerbalizED labels and verbalizations).")

    if len(verbalizations_dict) > len(gold_label_dict):
        print(f"Warning: The verbalizations dict ({len(verbalizations_dict)}) is larger than the actually used gold label dict ({len(gold_label_dict)}).")
        print("Only the gold label dict will be used! Make sure that you load all labels (either with use_corpus_labels or labels_path) or with replace_labels = False (adding the VerbalizED / ZELDA labels).")

    known_label_dict = gold_label_dict

    token_encoder = TransformerWordEmbeddings(
        model=transformer,
        fine_tune=True,
        use_context=context_size,
        layers="-1",
        context_dropout=0.0,
        allow_long_sentences=True,
    )

    if pooling in ["first_last", "mean"]:
        label_encoder = TransformerWordEmbeddings(
            model=transformer,
            use_context=False,
            fine_tune=True,
            layers="-1",
        )

    else:
        if pooling in ["first", "last"]:
            label_pooling = "cls"
        else:
            label_pooling = pooling
        label_encoder = TransformerDocumentEmbeddings(
            model=transformer, fine_tune=True, layers="-1", cls_pooling=label_pooling, use_context=False
        )  # called "mean" in transformer but "average" in DualEncoder


    kwargs = {
        "token_encoder": token_encoder,
        "label_encoder": label_encoder,
        "known_labels": known_label_dict.get_items(),
        "gold_labels": gold_label_dict.get_items(),
        "embedding_pooling": pooling,
        "label_type": "nel",
        "label_map": verbalizations_dict,
        "negative_sampling_factor": negative_sampling_factor,
        "negative_sampling_strategy": negative_sampling_strategy,
        "loss_function_name": loss,
        "similarity_metric_name": similarity_metric,
        "label_sample_negative_size": label_sample_size,
        "label_embedding_batch_size": label_embedding_batch_size,
        "constant_updating": constant_updating,
        "insert_in_context": insert_in_context,
        "insert_which_labels": insert_which_labels,
    }

    if greedy:
        model_class = GreedyDualEncoderEntityDisambiguation
    else:
        model_class = DualEncoderEntityDisambiguation

    model = model_class(**kwargs)

    trainer = ModelTrainer(model, corpus)

    weight_decay = False

    if not MULTI_GPU:
        trainer.fine_tune(
            model_path,
            max_epochs=epochs,
            learning_rate=learning_rate,
            mini_batch_size=batch_size,
            mini_batch_chunk_size=batch_size, #4
            eval_batch_size=batch_size_eval,
            #monitor_test=True,
            monitor_test=False,
            shuffle=True,
            shuffle_first_epoch=True,
            save_model_each_k_epochs=1,
            save_optimizer_state=True,
            epoch=epoch if continue_from_checkpoint else 0,
            attach_default_scheduler=False if continue_from_checkpoint else True,
            weight_decay = 0.1 if weight_decay else 0.01,
        )

    else:
        trainer.fine_tune(
            model_path,
            max_epochs=epochs,
            learning_rate=learning_rate,
            mini_batch_size=batch_size,
            mini_batch_chunk_size=batch_size, #4
            eval_batch_size=batch_size_eval,
            #monitor_test=True,
            monitor_test=False,
            shuffle=True,
            shuffle_first_epoch=True,
            save_model_each_k_epochs=1,
            save_optimizer_state=True,
            epoch=epoch if continue_from_checkpoint else 0,
            attach_default_scheduler=False if continue_from_checkpoint else True,
            weight_decay = 0.1 if weight_decay else 0.01,
            multi_gpu=True
        )

    print("That was:")
    print(model_path)

    return model_path


if __name__ == "__main__":
    from argparse import ArgumentParser
    argParser = ArgumentParser()
    argParser.add_argument("--config",
                           nargs="?",
                           default="../models/custom_dual_encoder/config.ini")
    args = argParser.parse_args()
    #args.config = "../models/custom_dual_encoder/config.ini" # TODO

    parser = ConfigParser()
    CONFIG_ROOT = Path("../models/custom_dual_encoder/")
    config_path = CONFIG_ROOT / args.config
    if config_path.is_file():
        parser.read(config_path)
    elif Path(args.config).is_file():
        parser.read(Path(args.config))
    else:
        raise FileNotFoundError(f"Config file not found: {config_path} or {args.config}")

    training_args = {
        "outdir": parser.get("General", "outdir"),
        "model_name": parser.get("General", "model_name"),
        "suffix": parser.get("General", "suffix", fallback="run"),
        "corpus_name": parser.get("Data", "corpus_name"),
        "label_type": parser.get("Data", "label_type", fallback="nel"),
        "document_level": parser.getboolean("Data", "document_level", fallback=False),
        "downsample_factor": parser.getfloat("Data", "downsample_factor", fallback=None),
        "verbalizations_path": parser.get("Data", "verbalizations_path", fallback=None),
        "labels_path": parser.get("Data", "labels_path", fallback=None),
        "use_corpus_labels": parser.getboolean("Data", "use_corpus_labels", fallback=False),
        "replace_labels": parser.getboolean("Data", "replace_labels", fallback=True),
        "replace_verbalizations": parser.getboolean("Data", "replace_verbalizations", fallback=True),
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

    try:
        training_args["negative_sampling_factor"] = int(training_args["negative_sampling_factor"])
    except:
        training_args["negative_sampling_factor"] = training_args["negative_sampling_factor"]

    if training_args["greedy"] == False:
        training_args.pop("insert_in_context", None)
        training_args.pop("insert_which_labels", None)

    if training_args["labels_path"] == "None":
        training_args["labels_path"] = None
    if training_args["verbalizations_path"] == "None":
        training_args["verbalizations_path"] = None


    print("Training with parameters:")
    for key, value in training_args.items():
        print(f"  {key:<25}: {value}")

    model_path = Path(training_args["outdir"]) / training_args["model_name"] / training_args["suffix"]
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # save the used config in the model path, add the final model path
    parser.set("General", "model_path", str(model_path))
    config_file = model_path / "config.ini"
    with open(config_file, "w") as f:
        parser.write(f)

    if not MULTI_GPU:
        train(**training_args)
    else:
        launch_distributed(train, **training_args) # todo

    print("Finished!")

