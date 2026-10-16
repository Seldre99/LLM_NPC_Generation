from __future__ import annotations

import os
import glob
from collections import defaultdict

import pandas as pd

INPUT_BASE = "analysis/writer/qwen"
OUTPUT_BASE = "analysis/distribution/writer/qwen"
MODEL_NAME = "McGill-NLP_AfriqueQwen-8B"

PATTERN = os.path.join(INPUT_BASE, "**", "*.csv")


def parse_csv_path(path: str) -> dict | None:
    parts = os.path.normpath(path).split(os.sep)
    try:
        base_idx = parts.index("writer")
    except ValueError:
        return None

    rel = parts[base_idx + 2 :]
    if len(rel) < 7:
        return None

    return {
        "modello": rel[0],
        "temp": rel[1],
        "genere_vg": rel[3],
        "genere_npc": rel[4],
        "body_type": rel[5],
        "etnia": os.path.splitext(rel[6])[0],
    }


def build_output_path(meta: dict) -> str:
    out_dir = os.path.join(
        OUTPUT_BASE,
        meta["modello"],
        meta["temp"],
        "english",
        meta["genere_vg"],
        meta["genere_npc"],
        meta["body_type"],
    )
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{meta['etnia']}.csv")


def process_csv(
    path: str,
    chosen_stats: list,
    role_dist: dict,
    role_dist_body: dict,
) -> None:
    meta = parse_csv_path(path)
    if meta is None:
        print(f"Path non valido, skip: {path}")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f" Impossibile leggere {path}: {e}")
        return

    if df.empty or "role" not in df.columns or "count" not in df.columns:
        print(f" Struttura CSV non valida, skip: {path}")
        return

    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

    total = df["count"].sum()
    if total == 0:
        print(f"  Tutti i conteggi sono 0, skip: {path}")
        return

    df["distribution"] = df["count"] / total

    # ── Statistiche "ruoli scelti" raggruppate per genere_vg ───────────────
    chosen_stats.append(
        {
            "genere_vg": meta["genere_vg"],
            "n_roles_total": len(df),
            "n_roles_chosen": int((df["count"] > 0).sum()),
        }
    )

    # ── Distribuzioni raggruppate per (genere_vg, genere_npc, etnia) ───────
    key = (meta["genere_vg"], meta["genere_npc"], meta["etnia"])

    for _, row in df.iterrows():
        role_dist[key][row["role"]].append(row["distribution"])

    # ── NUOVO:
    #    Distribuzioni raggruppate per
    #    (genere_vg, genere_npc, body_type) ──────────────────────────────────
    key_body = (meta["genere_vg"], meta["genere_npc"], meta["body_type"])

    for _, row in df.iterrows():
        role_dist_body[key_body][row["role"]].append(row["distribution"])

    out_path = build_output_path(meta)

    df[["role", "count", "distribution"]].to_csv(out_path, index=False)

    print(f" {out_path}")


def main() -> None:
    model_input_dir = os.path.join(INPUT_BASE, MODEL_NAME)

    csv_files = glob.glob(os.path.join(model_input_dir, "**", "*.csv"), recursive=True)

    if not csv_files:
        print(f"Nessun CSV trovato: {PATTERN}")
        return

    print(f"Trovati {len(csv_files)} file CSV.\n")

    chosen_stats = []

    # ── Esistente:
    #    (genere_vg, genere_npc, etnia)
    role_dist = defaultdict(lambda: defaultdict(list))

    # ── NUOVO:
    #    (genere_vg, genere_npc, body_type)
    role_dist_body = defaultdict(lambda: defaultdict(list))

    for path in sorted(csv_files):
        process_csv(
            path,
            chosen_stats,
            role_dist,
            role_dist_body,
        )

    print("\n Distribuzioni completate.")

    if not chosen_stats:
        return

    # ── Riepilogo:
    #    ruoli scelti almeno una volta, per genere_vg ────────────────────────
    chosen_df = pd.DataFrame(chosen_stats)

    chosen_df["ratio"] = chosen_df["n_roles_chosen"] / chosen_df["n_roles_total"]

    print(f"\n Statistiche modello: {MODEL_NAME}")

    print("\nRuoli scelti almeno una volta (media), " "per genere videoludico:")

    for genere_vg, group in chosen_df.groupby("genere_vg"):
        print(
            f"  {genere_vg}: "
            f"{group['ratio'].mean() * 100:.1f}% "
            f"({group['n_roles_chosen'].mean():.1f} su "
            f"{group['n_roles_total'].mean():.1f} "
            f"ruoli disponibili)"
        )

    # ── Riepilogo esistente:
    #    top 5 ruoli per
    #    (genere_vg, genere_npc, etnia) ─────────────────────────────────────
    print(
        "\nTop 5 ruoli (distribuzione media), "
        "per genere videoludico, genere NPC ed etnia:"
    )

    for (genere_vg, genere_npc, etnia), roles in sorted(role_dist.items()):

        avg_per_role = {role: sum(vals) / len(vals) for role, vals in roles.items()}

        top5 = sorted(avg_per_role.items(), key=lambda x: x[1], reverse=True)[:5]

        etnia_display = etnia.replace("_", " ").title()

        print(f"\n  {genere_vg} — " f"{genere_npc.capitalize()} — " f"{etnia_display}:")

        for role, avg_d in top5:
            print(f"    {role:<30} " f"{avg_d * 100:.1f}%")

    # ── NUOVO Riepilogo:
    #    top 5 ruoli per
    #    (genere_vg, genere_npc, body_type) ─────────────────────────────────
    print(
        "\nTop 5 ruoli (distribuzione media), "
        "per genere videoludico, genere NPC e body type:"
    )

    for (genere_vg, genere_npc, body_type), roles in sorted(role_dist_body.items()):

        avg_per_role = {role: sum(vals) / len(vals) for role, vals in roles.items()}

        top5 = sorted(avg_per_role.items(), key=lambda x: x[1], reverse=True)[:5]

        body_display = body_type.replace("_", " ").title()

        print(f"\n  {genere_vg} — " f"{genere_npc.capitalize()} — " f"{body_display}:")

        for role, avg_d in top5:
            print(f"    {role:<30} " f"{avg_d * 100:.1f}%")


if __name__ == "__main__":
    main()
