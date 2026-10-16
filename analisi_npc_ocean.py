"""
Analizza i JSON in:
  output/npc_quest/<modello>/<temperatura>/english/<genere_videogioco>/<genere_npc>/<body_type>/<etnia>.json

Struttura JSON di input: array di oggetti con "role" e "personality"
  (openness, conscientiousness, extraversion, agreeableness, neuroticism → high/medium/low)

Per ogni file JSON (= un'etnia), genera un CSV per ogni ruolo trovato:
  - righe: i 5 tratti OCEAN
  - colonne: high, medium, low (conteggi cumulativi)

Output salvato in:
  ocean/npc_quest/<modello>/<temperatura>/english/<genere_vg>/<genere_npc>/<body_type>/<etnia>/<ruolo>.csv
"""

import os
import glob
import json
from collections import defaultdict

import pandas as pd


INPUT_BASE  = "output/writer/qwen"
OUTPUT_BASE = "analysis/ocean/writer/qwen"
MODEL_NAME = "McGill-NLP_AfriqueQwen-8B"

PATTERN = os.path.join(INPUT_BASE, "**", "*.json")

OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
LEVELS       = ["high", "medium", "low"]


def build_output_dir(input_path: str) -> str:
    """
    Mappa il percorso input → cartella output, estraendo le parti significative.

    Struttura attesa dopo 'npc_quest':
      [modello, temperatura, english, genere_vg, genere_npc, body_type, <etnia>.json]
    """
    parts = os.path.normpath(input_path).split(os.sep)

    try:
        base_idx = parts.index("writer")
    except ValueError:
        raise ValueError(f" non trovato nel percorso: {input_path}")

    rel = parts[base_idx + 1:]

    if len(rel) < 7:
        raise ValueError(
            f"Struttura cartelle inattesa ({len(rel)} parti, attese ≥7): {rel}"
        )

    modello     = rel[1]
    temperatura = rel[2]
    # rel[2] = "english"
    genere_vg   = rel[4]
    genere_npc  = rel[5]
    body_type   = rel[6]
    # rel[6] = <etnia>.json  → diventa la sottocartella
    etnia       = os.path.splitext(rel[7])[0]  # rimuove .json

    print(etnia)

    etnia = etnia.split("_")[3]
    print(etnia)


    return os.path.join(
        OUTPUT_BASE,
        modello,
        temperatura,
        "english",
        genere_vg,
        genere_npc,
        body_type,
        etnia,
    )


def process_json(input_path: str) -> None:
    print(f"\n→ Processo: {input_path}")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Impossibile leggere il file: {e}")
        return

    if not isinstance(data, list) or len(data) == 0:
        print(" JSON vuoto o non è un array, skip.")
        return

    try:
        out_dir = build_output_dir(input_path)
    except ValueError as e:
        print(f"  ✗ {e}")
        return

    # Accumula conteggi: { role: { trait: { level: count } } }
    counts: dict = defaultdict(lambda: {trait: defaultdict(int) for trait in OCEAN_TRAITS})

    skipped = 0
    for entry in data:
        role        = entry.get("role", "").strip().title()
        personality = entry.get("personality", {})

        if not role or not personality:
            skipped += 1
            continue

        for trait in OCEAN_TRAITS:
            level = personality.get(trait, "").strip().lower()
            if level in LEVELS:
                counts[role][trait][level] += 1
            else:
                skipped += 1

    if skipped:
        print(f"  ⚠ {skipped} valori saltati (mancanti o non validi)")

    os.makedirs(out_dir, exist_ok=True)

    for role, trait_data in sorted(counts.items()):
        # Costruisce DataFrame: righe=tratti, colonne=high/medium/low
        rows = []
        for trait in OCEAN_TRAITS:
            rows.append({
                "trait":  trait,
                "high":   trait_data[trait].get("high",   0),
                "medium": trait_data[trait].get("medium", 0),
                "low":    trait_data[trait].get("low",    0),
            })

        df = pd.DataFrame(rows, columns=["trait", "high", "medium", "low"])

        # Nome file: ruolo sanitizzato
        safe_role = role.replace(" ", "_").replace("/", "-")
        out_path  = os.path.join(out_dir, f"{safe_role}.csv")

        df.to_csv(out_path, index=False)
        print(f" Salvato: {out_path}")


def main() -> None:
    model_input_dir = os.path.join(INPUT_BASE, MODEL_NAME)
    json_files = glob.glob(os.path.join(model_input_dir, "**", "*.json"), recursive=True)

    if not json_files:
        print(f"Nessun JSON trovato per il modello '{MODEL_NAME}' in: {model_input_dir}")
        print("Assicurati di eseguire lo script dalla directory che contiene 'output/'.")
        return

    print(f"Trovati {len(json_files)} file JSON da analizzare.")

    for json_path in sorted(json_files):
        process_json(json_path)

    print("\n Analisi completata.")


if __name__ == "__main__":
    main()
