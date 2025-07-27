import os
import time
import json
import requests
from tqdm import tqdm
from pathlib import Path

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
SUMMARY_ENDPOINT = "https://en.wikipedia.org/api/rest_v1/page/summary/"

HEADERS = {"User-Agent": "WikiVerbalizer"}

ROOT_PATH = Path("../data/verbalizations/")

WIKIDATA_PROPERTIES = {
    "P31": {"label": "instance of",   "ps": "ps:P31"},
    "P279": {"label": "subclass of",  "ps": "ps:P279"},
    "P361": {"label": "part of",      "ps": "ps:P361"},
    "P17": {"label": "country",       "ps": "ps:P17"},
    "P106": {"label": "occupation",   "ps": "ps:P106"},
}


def format_sparql_query(preferred_rank_entries_only: bool = True, qid: str = "Q31") -> str:
    """Formats a SPARQL query to retrieve structured information about a Wikidata item.
    :param preferred_rank_entries_only: bool: If True, only retrieves entries with preferred rank.
    :param qid: str: The Wikidata ID of the item to query (default is "Q31" for Belgium).
    :return: str: The formatted SPARQL query.
    """

    values_clause = "\n".join(
        f'(p:{pid} "{props["label"]}" {props["ps"]})'
        for pid, props in WIKIDATA_PROPERTIES.items()
    )

    filter_block = """
        FILTER(
          ?rank = wikibase:PreferredRank ||
          NOT EXISTS {
            ?id ?prop ?otherStatement .
            ?otherStatement wikibase:rank wikibase:PreferredRank ;
                            ?psProp ?_ .
          }
        )
    """ if preferred_rank_entries_only else ""

    query = f"""
    SELECT ?item ?propLabel ?itemLabel ?sitelinks ?outcoming ?idDescription
    WHERE {{
      BIND(wd:{qid} AS ?id)

      OPTIONAL {{
        ?id schema:description ?idDescription .
        FILTER(LANG(?idDescription) = "en")
      }}

      OPTIONAL {{
        VALUES (?prop ?propLabel ?psProp) {{
          {values_clause}
        }}

        ?id ?prop ?statement .
        ?statement ?psProp ?item .
        ?statement wikibase:rank ?rank .

        FILTER(?rank != wikibase:DeprecatedRank)

        {filter_block}

        OPTIONAL {{ ?item wikibase:statements ?outcoming. }}
        OPTIONAL {{ ?item wikibase:sitelinks ?sitelinks. }}

        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      }}
    }}
    GROUP BY ?item ?itemLabel ?propLabel ?sitelinks ?outcoming ?idDescription
    ORDER BY ?propLabel DESC(?sitelinks)
    LIMIT 50
    """
    return query.strip()



def execute_sparql(query: str, retries: int = 5, wait: int = 5) -> dict:
    """Executes a SPARQL query against the Wikidata endpoint with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.post(SPARQL_ENDPOINT, params={"query": query, "format": "json"}, headers=HEADERS)
            if response.status_code == 200:
                return response.json()
            elif response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(wait)
            else:
                break
        except Exception:
            time.sleep(wait)
    return {}


def get_one_paragraph_wikipedia_intro(title: str) -> str:
    """Fetches a one-paragraph summary of a Wikipedia article."""
    try:
        response = requests.get(SUMMARY_ENDPOINT + title, headers=HEADERS)
        if response.status_code == 200:
            return response.json().get("extract", "")
    except Exception:
        return None
    return None

def get_full_wikipedia_article_text(title: str, user_agent=None) -> str:
    """
    Fetches the full lead section (not just the first paragraph) of a Wikipedia article.
    Returns plain text (HTML stripped).
    """
    import requests
    from bs4 import BeautifulSoup

    headers = {"User-Agent": user_agent or "WikipediaIntroFetcher/1.0"}
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "redirects": True,
    }

    url = "https://en.wikipedia.org/w/api.php"

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        html_text = data["parse"]["text"]["*"]
        soup = BeautifulSoup(html_text, "html.parser")

        # Only get the content before the first section (lead)
        content_div = soup.find("div", {"class": "mw-parser-output"})
        paragraphs = content_div.find_all("p", recursive=False)

        # Join all non-empty <p> tags
        full_intro = "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text(strip=True))

        return full_intro

    except Exception as e:
        print(f"Error fetching lead for {title}: {e}")
        return None

def get_medium_wikipedia_intro(title: str, user_agent=None) -> str:
    """
    Fetches the lead section (intro paragraphs) of a Wikipedia article using MediaWiki extracts API.
    """
    import requests

    headers = {"User-Agent": user_agent or "WikipediaIntroFetcher/1.0"}
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,          # Only the lead section (before first heading)
        "explaintext": True,      # No HTML, plain text
        "redirects": True,
        "titles": title,
    }

    url = "https://en.wikipedia.org/w/api.php"

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract", "")
    except Exception as e:
        print(f"Error fetching medium intro for {title}: {e}")
        return None


def get_wikidata_id_and_shortdesc(title: str):
    """
    Retrieve the Wikidata ID and the short description of a Wikipedia article.

    :param title: Title of the Wikipedia article (e.g., "Belgium")
    :return: (wikidata_id, shortdesc) tuple, or (None, None) if not found
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "pageprops",
        "redirects": True
    }
    response = requests.get(url, params=params)

    try:
        page = next(iter(response.json()["query"]["pages"].values()))
        pageprops = page.get("pageprops", {})
        wikidata_id = pageprops.get("wikibase_item")
        shortdesc = pageprops.get("wikibase-shortdesc") or pageprops.get("shortdesc")
        return wikidata_id, shortdesc
    except Exception:
        return None, None


def extract_properties(result: dict) -> (list, str):
    seen = set()
    props = []
    description = None
    for r in result.get("results", {}).get("bindings", []):
        if "idDescription" in r and not description:
            description = r["idDescription"]["value"]
        if "itemLabel" in r and "propLabel" in r:
            label = r["itemLabel"]["value"]
            prop = r["propLabel"]["value"]
            if (prop, label) not in seen:
                seen.add((prop, label))
                props.append((prop, label))
    return props, description


def process_labels(label_list: list,
                   save_path: str = "verbalizations.json",
                   preferred_rank_entries_only: bool = True,
                   wikipedia_article_amount: str = "medium",
                   ):
    """
    Processes a list of Wikipedia labels (like ["Germany", "British_Museum"]) to gather structured information from Wikipedia and Wikidata.
    :param label_list: List of Wikipedia labels (e.g., ["Germany", "British_Museum"]).
    :param save_path: Path to save the results as a JSON file.
    :param preferred_rank_entries_only: If True, only fetches the preferred rank entries from Wikidata properties.
    :param wikipedia_article_amount: Specifies how much of the Wikipedia article to fetch. Must be one of "first", "medium", or "full".
    :return: A dictionary with labels as keys and their corresponding structured information as values.
    """

    allowed_amounts = ["first", "medium", "full"]
    if wikipedia_article_amount not in allowed_amounts:
        raise ValueError(
            f"`wikipedia_article_amount` must be one of {allowed_amounts}, got '{wikipedia_article_amount}'.")

    # ✅ Load existing results if save_path exists
    if os.path.exists(save_path):
        print(f"🔄 Loading existing data from {save_path}...")
        with open(save_path, "r", encoding="utf-8") as f:
            result_dict = json.load(f)
        print(f"✅ Found {len(result_dict)} existing entries. Continuing with remaining labels...")
    else:
        result_dict = {}

    skipped = 0

    for idx, label in enumerate(tqdm(label_list, desc="Processing labels")):
        if label in result_dict:
            skipped += 1
            continue  # Skip if already processed

        data = {
            "wikidata_id": None,
            "wikidata_url": None,
            "wikidata_properties": [],
            "wikidata_description": None,
            "wikipedia_paragraph": None
        }

        data["wikidata_title"] = label.replace("_", " ")
        wd_id, wikipedia_shortdescription = get_wikidata_id_and_shortdesc(label)
        if wd_id:
            data["wikidata_id"] = wd_id
            data["wikidata_url"] = f"https://www.wikidata.org/wiki/{wd_id}"
            data["wikipedia_description"] = wikipedia_shortdescription

            sparql_query = format_sparql_query(preferred_rank_entries_only=preferred_rank_entries_only, qid=wd_id)
            sparql_result = execute_sparql(sparql_query)
            props, desc = extract_properties(sparql_result)

            data["wikidata_properties"] = props
            data["wikidata_description"] = desc

        if wikipedia_article_amount == "first":
            summary = get_one_paragraph_wikipedia_intro(label)
        elif wikipedia_article_amount == "medium":
            summary = get_medium_wikipedia_intro(label)
        elif wikipedia_article_amount == "full":
            summary = get_full_wikipedia_article_text(label)

        data["wikipedia_paragraph"] = summary
        result_dict[label] = data

        # Save progress after certain number of entries
        if idx % 100 == 0 and idx > 0:
            with open(save_path, "w") as f:
                json.dump(result_dict, f, indent=2, sort_keys=True)

    # Save final results
    with open(save_path, "w") as f:
        json.dump(result_dict, f, indent=2, sort_keys=True)

    print(f"✅ Done. Skipped {skipped}/{len(label_list)} already-processed labels.")
    print(f"📁 Final results saved to: {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verbalize a list of Wikipedia labels.")
    parser.add_argument("--label_file", type=str, help="Path to a .txt file with one label per line.")
    parser.add_argument("--output", type=str, default="verbalizations.jsonl", help="Path to save the output JSON.")
    parser.add_argument("--amount", type=str, default="medium", choices=["first", "medium", "full"],
                        help="How much of the Wikipedia article to include: 'first', 'medium', or 'full'.")
    parser.add_argument("--no_preferred_only", action="store_true",
                        help="Include all of the properties' Wikidata statements, not just preferred-ranked ones.")

    args = parser.parse_args()

    if args.label_file:
        if not os.path.isfile(ROOT_PATH / args.label_file):
            raise FileNotFoundError(f"Label file '{ROOT_PATH / args.label_file}' not found.")
        with open(ROOT_PATH / args.label_file, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(labels)} labels from {ROOT_PATH / args.label_file} to verbalize!")
    else:
        # fallback example
        labels = [
            "Belgium", "Albert_Einstein", "British_Museum", "United_Kingdom",
        ]
        print("No label file provided. Using 4 demo labels.")

    process_labels(
        label_list=labels,
        save_path=ROOT_PATH / args.output,
        preferred_rank_entries_only=not args.no_preferred_only,
        wikipedia_article_amount=args.amount
    )


