import json
import itertools
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_quests(json_path: str):
    """Load aggregated quests JSON."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_dialogue(dialogue):
    """Convert dialogue (string or list of speech turns) into a single text string."""
    # Already simple string
    if isinstance(dialogue, str):
        return dialogue.strip()

    # Dialogue list with {speaker, text}
    if isinstance(dialogue, list):
        parts = []
        for d in dialogue:
            if isinstance(d, dict) and "text" in d:
                parts.append(d["text"])
        return " ".join(parts).strip()

    return ""


def compute_similarity(text1: str, text2: str, model) -> float:
    """Compute cosine similarity using embeddings."""
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    return float(cosine_similarity([emb1], [emb2])[0][0])


def analyze_similarity(quests: list, output_csv: str):
    """Compute pairwise similarity between all quests."""
    model = SentenceTransformer("all-MiniLM-L6-v2")

    results = []

    # Compare every pair of quests
    for q1, q2 in itertools.combinations(quests, 2):

        # normalize dialogue field
        dialog1 = normalize_dialogue(q1.get("dialogue", ""))
        dialog2 = normalize_dialogue(q2.get("dialogue", ""))

        name_sim = compute_similarity(q1.get("npc name", ""), q2.get("npc name", ""), model)
        quest_sim = compute_similarity(q1.get("quest", ""), q2.get("quest", ""), model)
        dialogue_sim = compute_similarity(dialog1, dialog2, model)

        total_sim = (name_sim + quest_sim + dialogue_sim) / 3

        results.append({
            "name1": q1.get("npc name", ""),
            "name2": q2.get("npc name", ""),
            "name_similarity": name_sim,
            "quest_similarity": quest_sim,
            "dialogue_similarity": dialogue_sim,
            "total_similarity": total_sim
        })

    df = pd.DataFrame(results)

    # Save CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\n✓ Similarity CSV saved: {output_csv}")

    # Summary statistics
    print("\n=== SIMILARITY SUMMARY ===")
    print("Npc name similarity avg:     ", df["name_similarity"].mean())
    print("Quest similarity avg:    ", df["quest_similarity"].mean())
    print("Dialogue similarity avg: ", df["dialogue_similarity"].mean())
    print("TOTAL similarity avg:    ", df["total_similarity"].mean())

    return df


def run_similarity(txt: str, model: str, lang: str, genre: str):
    """Main analysis function."""
    json_path = f"output/{txt}/{model.replace('/', '_')}/{lang}/quests_{lang}_{genre}.json"
    output_csv = f"similarity/{txt}/{model.replace('/', '_')}/{lang}/similarity_{genre}.csv"

    print(f"\n=== Loading quests: {json_path} ===")
    quests = load_quests(json_path)

    print(f"Loaded {len(quests)} quests, computing similarities...")

    analyze_similarity(quests, output_csv)


if __name__ == "__main__":
    # Example call:
    run_similarity("npc_quest", "google_gemma-3-1b", "english", "Contemporary Open World")
