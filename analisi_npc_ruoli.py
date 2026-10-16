"""
Analizza i CSV in:
  csv/npc_quest/<modello>/<temperatura>/english/<modello_usato>/english/<genere_videogioco>/<genere_npc>/<body_type>/<file>.csv

Per ogni CSV trovato, genera un CSV per ogni etnia presente, con i ruoli
ordinati dal più scelto al meno scelto, salvati in:
  analysis/npc_quest/<modello>/<temperatura>/english/<genere_videoludico>/<genere_npc>/<body_type>/<etnia>.csv

STRUTTURA ATTESA DEL CSV DI INPUT:
  - Prima colonna: nomi dei ruoli (o etichette di riga)
  - Colonne successive: una per ogni etnia, con i conteggi
  Esempio:
    role,Asian,Black,Hispanic,Middle Eastern,White
    Warrior,10,5,8,3,12
    Mage,4,9,2,7,6
    ...
"""

import os
import glob
import pandas as pd
import re


# ── Configurazione percorsi ───────────────────────────────────────────────────
INPUT_BASE  = "csv/writer_quest/qwen"
OUTPUT_BASE = "analysis/writer/qwen"
MODEL_NAME = "McGill-NLP_AfriqueQwen-8B"

# Profondità relativa delle cartelle dopo INPUT_BASE:
#   0: modello testato
#   1: temperatura
#   2: "english"        (fisso)
#   3: modello usato    (stesso livello, rinominato nella spec)
#   4: "english"        (fisso, secondo livello)
#   5: genere videogioco
#   6: genere npc
#   7: body type
#   → il CSV si trova qui dentro

PATTERN = os.path.join(INPUT_BASE, "**", "*.csv")


def build_output_path(input_path: str) -> str:
    """
    Mappa il percorso di input nel percorso di output,
    estraendo le parti significative dalla struttura di cartelle.
    """
    # Normalizza separatori
    parts = os.path.normpath(input_path).split(os.sep)

    # Trova l'indice di "npc_quest" nel percorso
    try:
        base_idx = parts.index("writer_quest")
    except ValueError:
        raise ValueError(f"'npc_quest' non trovato nel percorso: {input_path}")

    # Parti dopo "npc_quest":
    # [modello, temperatura, english, modello_usato, english, genere_vg, genere_npc, body_type, file.csv]
    rel_parts = parts[base_idx + 2:]  # esclude "npc_quest"

    if len(rel_parts) < 7:
        raise ValueError(
            f"Struttura cartelle inattesa (trovate {len(rel_parts)} parti, attese ≥7): {rel_parts}"
        )

    print(rel_parts)

    modello      = rel_parts[0]
    temperatura  = rel_parts[1]
    # rel_parts[2] = "english"
    genere_vg    = rel_parts[3]
    genere_npc   = rel_parts[4]
    body_type    = rel_parts[5]
    # rel_parts[6] = nome file csv (non usato, sostituito dall'etnia)

    # Output base per questo CSV:
    # analysis/npc_quest/<modello>/<temperatura>/english/<genere_vg>/<genere_npc>/<body_type>/
    out_dir = os.path.join(
        OUTPUT_BASE,
        modello,
        temperatura,
        "english",
        genere_vg,
        genere_npc,
        body_type,
    )
    return out_dir


def process_csv(input_path: str) -> None:
    print(f"\n→ Processo: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"  ✗ Impossibile leggere il file: {e}")
        return

    if df.empty or df.shape[1] < 2:
        print("  ✗ CSV vuoto o con meno di 2 colonne, skip.")
        return

    # Prima colonna = etnia; le altre = ruoli
    ethnicity_col = df.columns[0]
    eth_series = df[ethnicity_col]

    # ── Rimuovi colonne ruolo duplicate (stessa colonna ripetuta) ──────────
    role_df = df.iloc[:, 1:].copy()

    # Rimuove i suffissi aggiunti da pandas (.1, .2, ...)
    role_df.columns = [
        re.sub(r"\.\d+$", "", c)
        for c in role_df.columns
    ]

    # Tiene solo la prima occorrenza di ogni colonna
    role_df = role_df.loc[:, ~role_df.columns.duplicated(keep="first")]

    df = pd.concat([eth_series, role_df], axis=1)

    role_cols = df.columns[1:]

    try:
        out_dir = build_output_path(input_path)
    except ValueError as e:
        print(f"  ✗ {e}")
        return

    os.makedirs(out_dir, exist_ok=True)

    for _, row in df.iterrows():
        ethnicity = str(row[ethnicity_col]).strip()

        counts = pd.to_numeric(row[role_cols], errors="coerce").fillna(0)
        sorted_roles = (
            counts
            .sort_values(ascending=False)
            .reset_index()
        )
        sorted_roles.columns = ["role", "count"]
        sorted_roles["count"] = sorted_roles["count"].astype(int)

        safe_name = ethnicity.replace(" ", "_").replace("/", "-")
        out_path = os.path.join(out_dir, f"{safe_name}.csv")

        sorted_roles.to_csv(out_path, index=False)
        print(f"  ✔ Salvato: {out_path}")


def main() -> None:
    model_input_dir = os.path.join(INPUT_BASE, MODEL_NAME)
    csv_files = glob.glob(os.path.join(model_input_dir, "**", "*.csv"), recursive=True)

    if not csv_files:
        print(f"Nessun CSV trovato con il pattern: {PATTERN}")
        print("Assicurati di eseguire lo script dalla directory che contiene 'csv/'.")
        return

    print(f"Trovati {len(csv_files)} file CSV da analizzare.")

    for csv_path in sorted(csv_files):
        process_csv(csv_path)

    print("\nAnalisi completata.")


if __name__ == "__main__":
    main()
