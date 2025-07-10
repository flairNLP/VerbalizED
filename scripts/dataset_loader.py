from flair.datasets import NEL_ENGLISH_AIDA, ColumnCorpus, ZELDA
from pathlib import Path

def infer_and_load_dataset(user_input,
                           document_level=True,
                           dataset_path=Path("../data/datasets"),
                           column_format = {0: "text", 1: "nel"},
                           label_type = "nel",
                           only_test = False):

    if user_input == "zelda":
        label_type = "nel" # define under which "column" the labels are found
        if only_test:
            # Version 1: Load the current ZELDA corpus
            # Problem: This is dynamic, but it takes a long time to load the full dataset (including train, whchich we don't need)
            # corpus = ZELDA(document_level=document_level, in_memory=True)
            # corpus.train.dataset, corpus.dev.dataset = None, None
            # corpus.train.indices, corpus.dev.indices = [], []
            # return {d.path_to_column_file.stem: {"data": d, "label_type": label_type} for d in corpus.test.datasets}

            # Version 2: Load the local test files from the zelda_test folder, quicker
            folder = dataset_path / "zelda_test"  # → Path("../data/datasets/zelda_test")
            rt = {}
            for test_file in folder.iterdir():
                print("Reading in:", test_file.name)
                corpus = ColumnCorpus(folder,
                                      test_file=test_file.name,
                                      column_format=column_format,
                                      autofind_splits=False,
                                      sample_missing_splits=False,
                                      column_delimiter="\t",
                                      document_level=document_level,
                                      document_separator_token="-DOCSTART-",
                                      in_memory=True,
                                      )
                rt[test_file.stem] = {"data": corpus.test, "label_type": label_type}
            return rt

        else:
            corpus = ZELDA(document_level=document_level, in_memory=True)

            return {"zelda": {"data": corpus, "label_type": label_type}}

    elif user_input == "aida":
        corpus = NEL_ENGLISH_AIDA(document_level=document_level, use_ids_and_check_existence=True)
        label_type = "nel"
        if only_test:
            corpus.train.dataset, corpus.dev.dataset = None, None
            corpus.train.indices, corpus.dev.indices = [], []
            return {"aida": {"data": corpus.test, "label_type": label_type}}
        else:
            return {"aida": {"data": corpus, "label_type": label_type}}

    else:
        # check if user_input is a path
        if Path(user_input).exists():
            folder = Path(user_input).parent  # → Path("/path/to")
            filename = Path(user_input).name  # → "test.tsv"

        else:
            folder = dataset_path # → Path("../data/datasets")
            filename = user_input
            full_path = folder / filename

            # check if it's under the zelda_test folder
            if not full_path.exists():
                folder = dataset_path / "zelda_test"  # → Path("../data/datasets/zelda_test")
                filename = f"test_{user_input}.conll" # → "cweb" --> "test_cweb.conll"

        if only_test:
            corpus = ColumnCorpus(folder,
                                  test_file=filename,
                                  column_format=column_format,
                                  autofind_splits=False,
                                  sample_missing_splits=False,
                                  column_delimiter="\t",
                                  document_level=document_level,
                                  document_separator_token="-DOCSTART-",
                                  in_memory=True,
                                  )
        else:
            corpus = ColumnCorpus(folder / filename,
                                  column_format=column_format,
                                  autofind_splits=True,
                                  sample_missing_splits=False,
                                  column_delimiter="\t",
                                  document_level=document_level,
                                  document_separator_token="-DOCSTART-",
                                  in_memory=True,
                                  )

        label_type = label_type

        if only_test:
            return {Path(filename).stem: {"data": corpus.test, "label_type": label_type}}
        else:
            return {Path(filename).stem: {"data": corpus, "label_type": label_type}}
