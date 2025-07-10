from pathlib import Path
import json


def load_labels_from_file(path):
    """Loads labels from a .txt file, one label per line."""
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            label = line.strip()
            if label:
                labels.append(label)
    return labels

def custom_cutoff(string, initial_chars=50, max_extra_chars=50):
    # Find initial cutoff point by characters
    initial_cutoff = min(len(string), initial_chars)

    # Find the last space within the initial cutoff to avoid breaking words
    # last_space_initial = string.rfind(' ', 0, initial_cutoff)
    # if last_space_initial == -1:
    #    last_space_initial = initial_cutoff  # No space found, cut off at initial_chars

    # Define the search range for extra characters
    end_search = min(len(string), initial_cutoff + max_extra_chars)

    # Search for semicolon within the range
    semicolon_pos = string.find(";", initial_cutoff, end_search)
    if semicolon_pos != -1:
        return string[:semicolon_pos]

    # Search for comma within the range if no semicolon found
    comma_pos = string.find(",", initial_cutoff, end_search)
    if comma_pos != -1:
        return string[:comma_pos]

    # Take the whole string if it's inside the soft margin
    if len(string) <= initial_chars + max_extra_chars:
        return string

    # Only if no cut before found: Search for the last space within the range if no semicolon or comma found
    last_space = string.rfind(" ", initial_cutoff, end_search)
    if last_space != -1:
        return string[:last_space]

    # If neither found, cut off at the end of the search range
    return string[:end_search]

def normalize_verbalization_strategy(strategy):
    """Normalizes the verbalization strategy to a set of known strategies."""
    strategy_parts = strategy.lower().split("+")
    normalized_strategy = set()
    for part in strategy_parts:
        if part in {"description", "descriptions"}:
            normalized_strategy.add("description")
        elif part in {"class", "classes", "categories", "category"}:
            normalized_strategy.add("classes")
        elif part in {"title", "titles"}:
            normalized_strategy.add("title")
        elif part == "paragraph":
            normalized_strategy.add("paragraph")
        else:
            raise ValueError(f"Unknown verbalization strategy part: '{part}'")
    return normalized_strategy

def read_in_verbalizations_and_do_cutoff(path,
                                         verbalization_strategy,
                                         max_verbalization_length_soft):
    verbalizations_dict = {}

    with open(path, "r", encoding="utf-8") as file:
        p_dict = json.load(file)
        verbalizations_dict.update(p_dict)

    print(f"Loading a verbalization dict with {len(verbalizations_dict)} items.")

    # Parse and normalize strategy
    normalized_strategy = normalize_verbalization_strategy(verbalization_strategy)

    print(f"Using strategy: {', '.join(normalized_strategy)}")
    print(f"Applying soft cutoff at {max_verbalization_length_soft} characters...")

    for key, value in verbalizations_dict.items():
        verbalization = []
        classes_added = False

        if "title" in normalized_strategy:
            title = value.get("wikidata_title", key.replace("_", " "))
            if not title:
                title = key.replace("_", " ")
            verbalization.append(title)

        if "paragraph" in normalized_strategy:
            paragraph = value.get("wikipedia_paragraph")
            if paragraph:
                if verbalization:
                    verbalization.extend(["; ", paragraph])
                else:
                    verbalization.append(paragraph)

        if "description" in normalized_strategy:
            description = value.get("wikidata_description") or value.get("wikipedia_description")
            if description:
                verbalization.extend(["; ", description])
            else:
                # fallback: use classes if no description
                if "classes" in normalized_strategy:
                    classes = value.get("wikidata_properties", [])
                    for p, v in classes:
                        verbalization.extend(["; ", p, ": ", v])
                    classes_added = True

        if "classes" in normalized_strategy and not classes_added:
            classes = value.get("wikidata_properties", [])
            for p, v in classes:
                verbalization.extend(["; ", p, ": ", v])

        # Finalize verbalization with cutoff
        verbalization = "".join(verbalization)
        verbalizations_dict[key] = custom_cutoff(verbalization, initial_chars=max_verbalization_length_soft)

    return verbalizations_dict

