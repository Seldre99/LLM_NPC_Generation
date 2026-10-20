from __future__ import annotations
import re
from pathlib import Path
from itertools import product

import pandas as pd


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path("analysis/roles/statistical/writer/qwen/tokyotech-llm_Qwen3-Swallow-8B-CPT-v0.2")

TARGET_FILES = [
    "Contemporary Open World.csv",
    "Fantasy rpg.csv",
]

GENDERS = ["man", "woman", "non-binary"]
BODY_TYPES = ["skinny",
      "slim",
      "slender",
      "muscular",
      "overweight",
      "obese"]
ETHNICITIES = ["black person",
      "caucasian",
      "hispanic",
      "african american",
      "asian"]

ALL_LABEL = "🌐 all"

# Se i delta nel CSV sono tipo 0.20, diventano +20.
# Se sono già tipo 20, restano 20.
AUTO_PERCENT_SCALE = True


# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================

def normalize_colname(col: str) -> str:
    return (
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {normalize_colname(c): c for c in df.columns}

    for candidate in candidates:
        key = normalize_colname(candidate)
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Colonna non trovata. Cercavo una tra: {candidates}\n"
        f"Colonne disponibili: {list(df.columns)}"
    )


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Prova a riconoscere automaticamente le colonne principali.
    Modifica i candidate names se i tuoi CSV hanno nomi diversi.
    """

    gender_col = find_column(df, [
        "NPC", "npc"
    ])

    body_col = find_column(df, [
        "body_type", "body type", "body", "bodytype"
    ])

    ethnicity_col = find_column(df, [
        "ethnicity", "ethnic", "race"
    ])

    role_col = find_column(df, [
        "role", "occupation", "npc_role", "job", "class"
    ])

    delta_col = find_column(df, [
        "delta",
        "diff",
        "difference",
        "delta_probability",
        "probability_delta",
        "delta_prob",
        "delta_percentage",
        "delta_%",
        "Δ"
    ])

    return {
        "gender": gender_col,
        "body_type": body_col,
        "ethnicity": ethnicity_col,
        "role": role_col,
        "delta": delta_col,
    }


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_category(x) -> str:
    return clean_text(x).lower()


def clean_delta(x) -> float:
    """
    Gestisce delta scritti come:
    +20
    -13.5
    20%
    +0.20
    """
    if pd.isna(x):
        return 0.0

    s = str(x).strip()
    s = s.replace("%", "")
    s = s.replace(",", ".")

    match = re.search(r"[-+]?\d*\.?\d+", s)
    if not match:
        return 0.0

    return float(match.group())


def maybe_scale_delta(series: pd.Series) -> pd.Series:
    if not AUTO_PERCENT_SCALE:
        return series

    max_abs = series.abs().max()

    if pd.isna(max_abs):
        return series

    # Se tutti i valori sono tra -1 e +1, assumo che siano proporzioni.
    if max_abs <= 1:
        return series * 100

    return series


def format_delta(x: float) -> str:
    sign = "+" if x > 0 else ""
    if abs(x - round(x)) < 1e-9:
        return f"{sign}{int(round(x))}"
    return f"{sign}{x:.2f}"


def direction_icon(delta_min: float, delta_max: float) -> str:
    if delta_max < 0:
        return "🔴"
    if delta_min > 0:
        return "🟢"
    return "⚪"


# ============================================================
# COMPRESSIONE DEI PATTERN
# ============================================================

def block_size(block: tuple[str, str, str]) -> int:
    gender, body, ethnicity = block

    g_size = len(GENDERS) if gender == ALL_LABEL else 1
    b_size = len(BODY_TYPES) if body == ALL_LABEL else 1
    e_size = len(ETHNICITIES) if ethnicity == ALL_LABEL else 1

    return g_size * b_size * e_size


def expand_block(block: tuple[str, str, str]) -> set[tuple[str, str, str]]:
    gender, body, ethnicity = block

    genders = GENDERS if gender == ALL_LABEL else [gender]
    bodies = BODY_TYPES if body == ALL_LABEL else [body]
    ethnicities = ETHNICITIES if ethnicity == ALL_LABEL else [ethnicity]

    return set(product(genders, bodies, ethnicities))


def generate_valid_blocks(existing_combos: set[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """
    Genera blocchi validi del tipo:
    all, muscular, all
    man, obese, all
    all, all, black
    ecc.

    Un blocco è valido solo se tutte le combinazioni implicate sono presenti.
    """

    gender_options = GENDERS + [ALL_LABEL]
    body_options = BODY_TYPES + [ALL_LABEL]
    ethnicity_options = ETHNICITIES + [ALL_LABEL]

    valid_blocks = []

    for block in product(gender_options, body_options, ethnicity_options):
        expanded = expand_block(block)

        if expanded and expanded.issubset(existing_combos):
            valid_blocks.append(block)

    # Prima i blocchi più grandi, poi quelli più specifici.
    valid_blocks.sort(
        key=lambda b: (
            -block_size(b),
            b[1],  # body type
            b[0],  # gender
            b[2],  # ethnicity
        )
    )

    return valid_blocks


def compress_role_patterns(df_role: pd.DataFrame) -> pd.DataFrame:
    """
    Comprime le righe significative di un singolo ruolo e di un singolo segno.
    """

    combo_delta = {}

    for _, row in df_role.iterrows():
        combo = (
            row["Gender"],
            row["Body type"],
            row["Ethnicity"],
        )

        delta = row["Delta"]

        if combo not in combo_delta:
            combo_delta[combo] = [delta, delta]
        else:
            combo_delta[combo][0] = min(combo_delta[combo][0], delta)
            combo_delta[combo][1] = max(combo_delta[combo][1], delta)

    remaining = set(combo_delta.keys())
    rows = []

    while remaining:
        valid_blocks = generate_valid_blocks(remaining)

        if not valid_blocks:
            # fallback di sicurezza
            combo = sorted(remaining)[0]
            valid_blocks = [combo]

        block = valid_blocks[0]
        expanded = expand_block(block) & remaining

        deltas_min = [combo_delta[c][0] for c in expanded]
        deltas_max = [combo_delta[c][1] for c in expanded]

        delta_min = min(deltas_min)
        delta_max = max(deltas_max)

        rows.append({
            "Icon": direction_icon(delta_min, delta_max),
            "Gender": block[0],
            "Body type": block[1],
            "Ethnicity": block[2],
            "Role": df_role["Role"].iloc[0],
            "Delta min": delta_min,
            "Delta max": delta_max,
            "N combinations": len(expanded),
        })

        remaining -= expanded

    return pd.DataFrame(rows)


def build_synthetic_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la tabella sintetica completa.
    I positivi e i negativi vengono processati separatamente.
    """

    output_parts = []

    for sign_name, condition in [
        ("positive", df["Delta"] > 0),
        ("negative", df["Delta"] < 0),
    ]:
        df_sign = df[condition].copy()

        for role, df_role in df_sign.groupby("Role", sort=True):
            compressed = compress_role_patterns(df_role)
            compressed["Direction"] = sign_name
            output_parts.append(compressed)

    if not output_parts:
        return pd.DataFrame(columns=[
            "Icon", "Direction", "Gender", "Body type", "Ethnicity",
            "Role", "Delta min", "Delta max", "N combinations"
        ])

    result = pd.concat(output_parts, ignore_index=True)

    # Ordinamento per pattern vicini, non per delta assoluto.
    result = result.sort_values(
        by=["Direction", "Role", "Body type", "Gender", "Ethnicity"],
        ascending=[True, True, True, True, True]
    ).reset_index(drop=True)

    return result


# ============================================================
# ANALISI FILE
# ============================================================

def prepare_input_dataframe(csv_path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)
    cols = detect_columns(df_raw)

    df = pd.DataFrame({
        "Gender": df_raw[cols["gender"]].map(clean_category),
        "Body type": df_raw[cols["body_type"]].map(clean_category),
        "Ethnicity": df_raw[cols["ethnicity"]].map(clean_category),
        "Role": df_raw[cols["role"]].map(clean_text),
        "Delta": df_raw[cols["delta"]].map(clean_delta),
    })

    df["Delta"] = maybe_scale_delta(df["Delta"])

    # Rimuovo eventuali delta zero.
    df = df[df["Delta"] != 0].copy()

    return df


def create_role_summary(synthetic: pd.DataFrame) -> pd.DataFrame:
    if synthetic.empty:
        return pd.DataFrame()

    summary = (
        synthetic
        .groupby(["Direction", "Role"], as_index=False)
        .agg(
            delta_min=("Delta min", "min"),
            delta_max=("Delta max", "max"),
            n_patterns=("Role", "count"),
            n_combinations=("N combinations", "sum"),
        )
    )

    summary["Icon"] = summary.apply(
        lambda r: "🟢" if r["Direction"] == "positive" else "🔴",
        axis=1
    )

    summary = summary[[
        "Icon", "Direction", "Role",
        "delta_min", "delta_max",
        "n_patterns", "n_combinations"
    ]]

    return summary


def save_outputs(csv_path: Path, synthetic: pd.DataFrame, original: pd.DataFrame) -> None:
    output_dir = csv_path.parent / "synthetic_tables"
    output_dir.mkdir(exist_ok=True)

    stem = csv_path.stem.lower().replace(" ", "_")

    csv_complete = output_dir / f"{stem}_tabella_sintetica.csv"
    csv_positive = output_dir / f"{stem}_tabella_sintetica_delta_positivi.csv"
    csv_negative = output_dir / f"{stem}_tabella_sintetica_delta_negativi.csv"
    xlsx_path = output_dir / f"{stem}_tabella_sintetica.xlsx"

    positive = synthetic[synthetic["Direction"] == "positive"].copy()
    negative = synthetic[synthetic["Direction"] == "negative"].copy()
    role_summary = create_role_summary(synthetic)

    # Formattazione delta per i CSV finali leggibili.
    def readable(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not out.empty:
            out["Delta min"] = out["Delta min"].map(format_delta)
            out["Delta max"] = out["Delta max"].map(format_delta)
        return out

    readable(synthetic).to_csv(csv_complete, index=False, encoding="utf-8-sig")
    readable(positive).to_csv(csv_positive, index=False, encoding="utf-8-sig")
    readable(negative).to_csv(csv_negative, index=False, encoding="utf-8-sig")

    overview = pd.DataFrame([
        ["File", csv_path.name],
        ["Path", str(csv_path)],
        ["Test significativi analizzati", len(original)],
        ["Delta positivi", int((original["Delta"] > 0).sum())],
        ["Delta negativi", int((original["Delta"] < 0).sum())],
        ["Righe sintetiche finali", len(synthetic)],
        ["Righe sintetiche positive", len(positive)],
        ["Righe sintetiche negative", len(negative)],
    ], columns=["Metric", "Value"])

    legend = pd.DataFrame([
        ["🟢", "Delta positivo: il ruolo aumenta rispetto alla baseline"],
        ["🔴", "Delta negativo: il ruolo diminuisce rispetto alla baseline"],
        ["🌐 all", "La relazione vale per tutte le combinazioni disponibili di quella dimensione"],
        ["Delta min", "Delta minimo osservato nel blocco sintetico"],
        ["Delta max", "Delta massimo osservato nel blocco sintetico"],
        ["N combinations", "Numero di combinazioni originali coperte dalla riga sintetica"],
    ], columns=["Symbol", "Meaning"])

    print(f"Creati output per: {csv_path}")
    print(f"  CSV completo: {csv_complete}")
    print(f"  CSV positivi: {csv_positive}")
    print(f"  CSV negativi: {csv_negative}")
    print()


def analyze_file(csv_path: Path) -> None:
    original = prepare_input_dataframe(csv_path)
    synthetic = build_synthetic_table(original)
    save_outputs(csv_path, synthetic, original)


def main() -> None:
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Cartella non trovata: {BASE_DIR}")

    found_files = []

    for target_name in TARGET_FILES:
        found_files.extend(BASE_DIR.rglob(target_name))

    found_files = sorted(set(found_files))

    if not found_files:
        print("Nessun file trovato.")
        return

    print(f"Trovati {len(found_files)} file da analizzare.\n")

    for csv_path in found_files:
        try:
            analyze_file(csv_path)
        except Exception as e:
            print(f"Errore su file: {csv_path}")
            print(e)
            print()


if __name__ == "__main__":
    main()
