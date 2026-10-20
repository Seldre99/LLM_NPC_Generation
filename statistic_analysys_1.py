from __future__ import annotations
import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ── Configurazione ─────────────────────────────────────────────────────────────
INPUT_BASE = "analysis/roles/writer/qwen"
OUTPUT_BASE = "analysis/roles/statistical/writer/qwen"
BASELINE = "Qwen_Qwen3-8B"

ALPHA = 0.05
ASR_THRESHOLD = 2.58  # |ASR| minimo (2.58 ≈ p<0.01 a due code)
DELTA_THRESHOLD = 0.05  # |Δ| minimo (5 punti percentuali)

PATTERN = os.path.join(INPUT_BASE, "**", "*.csv")


# ── Parsing ────────────────────────────────────────────────────────────────────


def parse_role_path(path: str) -> dict | None:
    """
    analysis/roles/npc_quest/<genere_vg>/<genere_npc>/<body_type>/<etnia>/<role>.csv
    """
    parts = os.path.normpath(path).split(os.sep)
    try:
        base_idx = parts.index("writer")
    except ValueError:
        return None

    rel = parts[base_idx + 2 :]
    if len(rel) < 5:
        return None

    return {
        "genere_vg": rel[0],
        "genere_npc": rel[1],
        "body_type": rel[2],
        "etnia": rel[3],
        "role": os.path.splitext(rel[4])[0].replace("_", " "),
    }


# ── Caricamento ────────────────────────────────────────────────────────────────


def load_all_roles(csv_files: list[str]) -> dict:
    """
    Restituisce:
      data[(genere_vg, genere_npc, body_type, etnia, role)]
        = { modello: {"count": int, "other": int} }
    """
    data = defaultdict(dict)

    for path in csv_files:
        meta = parse_role_path(path)
        if meta is None:
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  {path}: {e}")
            continue

        if df.empty or not {"modello", "count", "other"}.issubset(df.columns):
            continue

        key = (
            meta["genere_vg"],
            meta["genere_npc"],
            meta["body_type"],
            meta["etnia"],
            meta["role"],
        )

        for _, row in df.iterrows():
            data[key][str(row["modello"])] = {
                "count": int(row["count"]),
                "other": int(row["other"]),
            }

    return data


# ── Adjusted Standardized Residual ─────────────────────────────────────────────


def adjusted_standardized_residual(
    base_count: int, base_other: int, model_count: int, model_other: int
) -> float:
    """
    ASR per la cella (modello, count) di una tabella 2×2.
    Positivo → il modello sceglie quel ruolo PIÙ della baseline.
    Negativo → il modello sceglie quel ruolo MENO della baseline.
    """
    table = np.array(
        [[base_count, base_other], [model_count, model_other]], dtype=float
    )

    n = table.sum()
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)

    if n == 0 or (row_sums == 0).any() or (col_sums == 0).any():
        return 0.0

    expected = np.outer(row_sums, col_sums) / n
    p_row = row_sums / n
    p_col = col_sums / n

    denom = np.sqrt(expected[1, 0] * (1 - p_row[1]) * (1 - p_col[0]))
    if denom == 0:
        return 0.0

    return float((table[1, 0] - expected[1, 0]) / denom)


# ── χ² per singolo ruolo ───────────────────────────────────────────────────────


def run_chi2_2x2(
    base_count: int, base_other: int, model_count: int, model_other: int
) -> dict | None:
    """
    Tabella di contingenza 2×2:
              count   other
    baseline    a       b
    model       c       d
    """
    table = np.array(
        [[base_count, base_other], [model_count, model_other]], dtype=float
    )

    if table.sum() == 0 or (table.sum(axis=1) == 0).any():
        return None

    try:
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    except Exception:
        return None

    p_base = (
        base_count / (base_count + base_other) if (base_count + base_other) > 0 else 0
    )
    p_model = (
        model_count / (model_count + model_other)
        if (model_count + model_other) > 0
        else 0
    )
    delta = p_model - p_base

    asr = adjusted_standardized_residual(
        base_count, base_other, model_count, model_other
    )

    return {
        "chi2": round(chi2, 4),
        "p_raw": p,
        "p_base": round(p_base, 4),
        "p_model": round(p_model, 4),
        "delta": round(delta, 4),
        "asr": round(asr, 4),
    }


# ── Analisi ────────────────────────────────────────────────────────────────────


def analyze(data: dict) -> None:
    """
    Raggruppa i confronti per (modello, genere_vg),
    applica BH, filtra su |ASR| e |Δ| e salva i risultati.
    """

    comparisons: dict = defaultdict(list)

    for (genere_vg, genere_npc, body_type, etnia, role), model_data in data.items():

        if BASELINE not in model_data:
            continue

        base = model_data[BASELINE]

        for modello, mdata in model_data.items():
            if modello == BASELINE:
                continue

            res = run_chi2_2x2(
                base["count"],
                base["other"],
                mdata["count"],
                mdata["other"],
            )
            if res is None:
                continue

            comparisons[(modello, genere_vg)].append(
                {
                    "genere_npc": genere_npc,
                    "body_type": body_type,
                    "etnia": etnia,
                    "role": role,
                    **res,
                }
            )

    # ── Per ogni (modello, genere_vg) applica BH + filtri e salva ─────────
    for (modello, genere_vg), records in sorted(comparisons.items()):
        df = pd.DataFrame(records)

        if df.empty:
            continue

        reject, p_adj, _, _ = multipletests(
            df["p_raw"].values, alpha=ALPHA, method="fdr_bh"
        )
        df["p_adj"] = np.round(p_adj, 6)
        df["significant"] = reject

        # ── Filtro triplo: adj-p < ALPHA, |ASR| > soglia, |Δ| > soglia ────
        df_sig = df[
            df["significant"]
            & (df["asr"].abs() > ASR_THRESHOLD)
            & (df["delta"].abs() > DELTA_THRESHOLD)
        ].copy()

        n_tot = len(df)
        n_sig = len(df_sig)
        print(
            f"  {modello} | {genere_vg}: {n_tot} confronti, {n_sig} dopo filtro completo"
        )

        if df_sig.empty:
            continue

        # Formatta Δ come percentuale con segno
        df_sig["delta_pct"] = df_sig["delta"].apply(
            lambda x: f"{'+' if x >= 0 else ''}{x*100:.1f}%"
        )

        # Ordina per |ASR| decrescente
        df_sig["abs_asr"] = df_sig["asr"].abs()
        df_sig = df_sig.sort_values("abs_asr", ascending=False).drop(
            columns=["abs_asr", "delta", "p_raw", "significant"]
        )

        # Rinomina colonne per il report finale
        df_out = df_sig.rename(
            columns={
                "genere_npc": "NPC",
                "body_type": "Body",
                "etnia": "Ethnicity",
                "role": "Role",
                "delta_pct": "Δ",
                "p_adj": "adj-p",
                "p_base": "p_baseline",
                "p_model": "p_model",
                "chi2": "chi2",
                "asr": "ASR",
            }
        )[
            [
                "NPC",
                "Body",
                "Ethnicity",
                "Role",
                "Δ",
                "adj-p",
                "ASR",
                "p_baseline",
                "p_model",
                "chi2",
            ]
        ]

        out_dir = os.path.join(OUTPUT_BASE, modello)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{genere_vg}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"  Salvato: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    csv_files = glob.glob(PATTERN, recursive=True)

    if not csv_files:
        print(f"Nessun CSV trovato: {PATTERN}")
        return

    print(f"Trovati {len(csv_files)} file CSV.\n")

    data = load_all_roles(csv_files)
    if not data:
        print("Nessun dato valido caricato.")
        return

    print(f"Ruoli caricati: {len(data)}\n")
    analyze(data)

    print("\n Analisi completata.")


if __name__ == "__main__":
    main()
