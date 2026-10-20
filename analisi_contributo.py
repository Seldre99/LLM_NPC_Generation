import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# ── Configurazione ─────────────────────────────────────────────────────────────

INPUT_BASE = "analysis/distribution"
OUTPUT_BASE = "analysis/influence/llama"

PERSPECTIVES = ["npc", "writer"]
FAMILY = "llama"
TEMP_FOLDER = "temperature_0.8"

PATTERNS = [
    os.path.join(INPUT_BASE, perspective, FAMILY, "**", "*.csv")
    for perspective in PERSPECTIVES
]


# ── Parsing ────────────────────────────────────────────────────────────────────


def parse_path(path: str) -> dict | None:
    """
    Struttura attesa:

    analysis/distribution/
        <perspective>/
        <family>/
        <modello>/
        temperature_0.8/
        english/
        <genere_vg>/
        <genere_npc>/
        <body>/
        <etnia>.csv

    Esempio:

    analysis/distribution/npc/gemma/google_gemma-2-9b/
        temperature_0.8/english/
        Contemporary Open World/man/skinny/asian.csv
    """

    parts = os.path.normpath(path).split(os.sep)

    try:
        dist_idx = parts.index("distribution")
    except ValueError:
        return None

    rel = parts[dist_idx + 1 :]

    # rel[0] = perspective
    # rel[1] = family
    # rel[2] = modello
    # rel[3] = temperature_0.8
    # rel[4] = english
    # rel[5] = genere_vg
    # rel[6] = genere_npc
    # rel[7] = body
    # rel[8] = etnia.csv

    if len(rel) < 9:
        return None

    if rel[0] not in PERSPECTIVES:
        return None

    if rel[1] != FAMILY:
        return None

    if rel[3] != TEMP_FOLDER:
        return None

    if rel[4] != "english":
        return None

    return {
        "perspective": rel[0],
        "family": rel[1],
        "modello": rel[2],
        "genere_vg": rel[5],
        "genere_npc": rel[6],
        "body": rel[7],
        "etnia": os.path.splitext(rel[8])[0],
    }


# ── Cramér's V ─────────────────────────────────────────────────────────────────


def cramers_v(col_a: pd.Series, col_b: pd.Series) -> float:
    """
    Cramér's V con correzione bias tra due variabili categoriali.

    Ogni riga viene considerata dopo l'espansione in base al Count.
    """

    contingency = pd.crosstab(col_a, col_b)

    n = contingency.values.sum()

    if n == 0:
        return 0.0

    try:
        chi2, _, _, _ = chi2_contingency(contingency, correction=False)
    except Exception:
        return 0.0

    r, k = contingency.shape

    phi2 = chi2 / n

    # Evita divisioni problematiche per campioni troppo piccoli
    if n <= 1:
        return 0.0

    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))

    k_corr = k - (k - 1) ** 2 / (n - 1)

    r_corr = r - (r - 1) ** 2 / (n - 1)

    denom = min(k_corr - 1, r_corr - 1)

    if denom <= 0:
        return 0.0

    return float(np.sqrt(phi2_corr / denom))


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:

    # ── Cerca solo i CSV della famiglia selezionata ────────────────────────

    csv_files = []

    for pattern in PATTERNS:
        csv_files.extend(glob.glob(pattern, recursive=True))

    if not csv_files:
        print(f"Nessun CSV trovato per la famiglia: {FAMILY}")
        return

    print(f"Trovati {len(csv_files)} file CSV " f"per la famiglia {FAMILY}.\n")

    # ── Raccolta dati ──────────────────────────────────────────────────────
    #
    # Struttura:
    #
    # {
    #   (perspective, modello, genere_vg): [righe]
    # }
    #
    # Esempio:
    #
    # ("npc", "google_gemma-2-9b", "Contemporary Open World")
    #

    buckets: dict = defaultdict(list)

    for path in sorted(csv_files):

        meta = parse_path(path)

        if meta is None:
            continue

        try:
            df = pd.read_csv(path)

        except Exception as e:

            print(f" {path}: {e}")

            continue

        if df.empty or "role" not in df.columns or "count" not in df.columns:
            continue

        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

        # Considera solo i ruoli effettivamente scelti
        df = df[df["count"] > 0]

        if df.empty:
            continue

        key = (meta["perspective"], meta["modello"], meta["genere_vg"])

        for _, row in df.iterrows():

            buckets[key].append(
                {
                    "Gender": meta["genere_npc"],
                    "Ethnicity": meta["etnia"],
                    "Body": meta["body"],
                    "Role": str(row["role"]).strip(),
                    "Count": int(row["count"]),
                }
            )

    if not buckets:

        print("Nessun dato valido caricato.")

        return

    # ── Per ogni bucket:
    #    salva flat CSV + calcola Cramér's V ───────────────────────────────

    cramer_summary = []

    for (perspective, modello, genere_vg), rows in sorted(buckets.items()):

        flat_df = pd.DataFrame(rows)

        # ── Espansione tramite Count ───────────────────────────────────────
        #
        # Ogni combinazione:
        #
        # Gender / Ethnicity / Body / Role
        #
        # viene ripetuta Count volte.
        #

        expanded = flat_df.loc[flat_df.index.repeat(flat_df["Count"])].reset_index(
            drop=True
        )

        # ── Directory output ───────────────────────────────────────────────

        out_dir = os.path.join(OUTPUT_BASE, modello)

        os.makedirs(out_dir, exist_ok=True)

        # ── Nome genere sicuro per file ───────────────────────────────────

        safe_genre = genere_vg.replace(" ", "_")

        # ── Salva flat CSV ────────────────────────────────────────────────

        flat_path = os.path.join(out_dir, f"{perspective}_{safe_genre}.csv")

        flat_df.to_csv(flat_path, index=False)

        print(f"  Flat CSV: " f"{flat_path} " f"({len(flat_df)} righe)")

        # ── Cramér's V:
        #    Gender vs Role
        #    Ethnicity vs Role
        #    Body vs Role ─────────────────────────────────────────────────

        v_gender = cramers_v(expanded["Gender"], expanded["Role"])

        v_ethnicity = cramers_v(expanded["Ethnicity"], expanded["Role"])

        v_body = cramers_v(expanded["Body"], expanded["Role"])

        # ── Conversione in percentuali relative ──────────────────────────

        total = v_gender + v_ethnicity + v_body

        if total == 0:

            pct_gender = 0.0
            pct_ethnicity = 0.0
            pct_body = 0.0

        else:

            pct_gender = v_gender / total * 100

            pct_ethnicity = v_ethnicity / total * 100

            pct_body = v_body / total * 100

        # ── Riga di riepilogo ────────────────────────────────────────────

        cramer_row = {
            "perspective": perspective,
            "family": FAMILY,
            "modello": modello,
            "genere_vg": genere_vg,
            "V_Gender": round(v_gender, 4),
            "V_Ethnicity": round(v_ethnicity, 4),
            "V_Body": round(v_body, 4),
            "pct_Gender": round(pct_gender, 1),
            "pct_Ethnicity": round(pct_ethnicity, 1),
            "pct_Body": round(pct_body, 1),
        }

        cramer_summary.append(cramer_row)

        # ── Salva risultato Cramér per singolo bucket ────────────────────

        cramer_path = os.path.join(out_dir, f"{perspective}_{safe_genre}_cramer.csv")

        pd.DataFrame([cramer_row]).to_csv(cramer_path, index=False)

        print(
            f"  Cramér:   {cramer_path}\n"
            f"  Gender={pct_gender:.1f}%  "
            f"Ethnicity={pct_ethnicity:.1f}%  "
            f"Body={pct_body:.1f}%\n"
        )

    # ── Riepilogo globale ──────────────────────────────────────────────────

    if cramer_summary:

        summary_df = pd.DataFrame(cramer_summary)

        summary_path = os.path.join(OUTPUT_BASE, "cramer_summary_all.csv")

        summary_df.to_csv(summary_path, index=False)

        print(f"\n Riepilogo globale: " f"{summary_path}")

        print("\n Influenza % per modello / " "genere videoludico / prospettiva:")

        print(
            f"  "
            f"{'Perspective':<12} "
            f"{'Modello':<45} "
            f"{'Genere VG':<30} "
            f"{'Gender':>8} "
            f"{'Ethnicity':>10} "
            f"{'Body':>8}"
        )

        print("  " + "─" * 120)

        for _, r in summary_df.iterrows():

            print(
                f"  "
                f"{r['perspective']:<12} "
                f"{r['modello']:<45} "
                f"{r['genere_vg']:<30}"
                f"  "
                f"{r['pct_Gender']:>6.1f}%  "
                f"{r['pct_Ethnicity']:>8.1f}%  "
                f"{r['pct_Body']:>6.1f}%"
            )


if __name__ == "__main__":
    main()
