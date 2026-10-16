from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Perspective to analyze:
# "npc" or "narrative"
PERSPECTIVE = "narrative"

# Model family
FAMILY = "qwen"

# Model to analyze
MODEL_NAME = "McGill-NLP_AfriqueQwen-8B"

INPUT_BASE = Path("analysis/ocean") / PERSPECTIVE / FAMILY / MODEL_NAME


OUTPUT_BASE = Path("analysis/ocean_distribution") / PERSPECTIVE / FAMILY / MODEL_NAME


OCEAN_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

LEVELS = [
    "high",
    "medium",
    "low",
]


# Genres expected in the folder structure
GENRES = {
    "Contemporary Open World": "contemporary_open_world",
    "Fantasy rpg": "fantasy_rpg",
}


# ============================================================
# EXTRACT GAME GENRE FROM PATH
# ============================================================


def extract_game_genre(
    csv_path: Path,
) -> str | None:

    try:
        relative = csv_path.relative_to(INPUT_BASE)
    except ValueError:
        return None

    parts = relative.parts

    # Expected:
    #
    # temperature /
    # english /
    # game_genre /
    # gender /
    # body_type /
    # ethnicity /
    # role.csv
    #
    # Therefore game genre is index 2.

    if len(parts) < 7:
        print(f" Unexpected path structure: " f"{csv_path}")
        return None

    return parts[2]


# ============================================================
# PROCESS SINGLE CSV
# ============================================================


def process_csv(
    csv_path: Path,
) -> list[dict]:

    try:
        df = pd.read_csv(csv_path)

    except Exception as exc:

        print(f" Cannot read " f"{csv_path}: {exc}")

        return []

    required_columns = {
        "trait",
        "high",
        "medium",
        "low",
    }

    if not required_columns.issubset(df.columns):

        print(f" Invalid CSV structure: " f"{csv_path}")

        return []

    game_genre = extract_game_genre(csv_path)

    if game_genre is None:
        return []

    df["trait"] = df["trait"].astype(str).str.strip().str.lower()

    for level in LEVELS:

        df[level] = pd.to_numeric(
            df[level],
            errors="coerce",
        ).fillna(0)

    results = []

    for trait in OCEAN_TRAITS:

        trait_df = df[df["trait"] == trait]

        if trait_df.empty:

            print(f" Missing trait '{trait}' " f"in {csv_path}")

            continue

        row = trait_df.iloc[0]

        high = float(row["high"])

        medium = float(row["medium"])

        low = float(row["low"])

        total = high + medium + low

        if total <= 0:

            print(f" Total = 0 for " f"{trait} in " f"{csv_path}")

            continue

        results.append(
            {
                "file": str(csv_path.relative_to(INPUT_BASE)),
                "game_genre": game_genre,
                "trait": trait,
                "high": high / total,
                "medium": medium / total,
                "low": low / total,
                "n": total,
            }
        )

    return results


# ============================================================
# COMPUTE MEAN DISTRIBUTION
# ============================================================


def compute_mean_distribution(
    genre_df: pd.DataFrame,
) -> pd.DataFrame:

    mean_df = genre_df.groupby("trait", as_index=False,).agg(
        high=(
            "high",
            "mean",
        ),
        medium=(
            "medium",
            "mean",
        ),
        low=(
            "low",
            "mean",
        ),
        n_csv=(
            "file",
            "nunique",
        ),
    )

    trait_order = {trait: index for index, trait in enumerate(OCEAN_TRAITS)}

    mean_df["_order"] = mean_df["trait"].map(trait_order)

    mean_df = (
        mean_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    )

    for level in LEVELS:

        mean_df[f"{level}_pct"] = mean_df[level] * 100

    return mean_df


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    if not INPUT_BASE.exists():

        print(f"Input directory " f"not found:\n" f"{INPUT_BASE}")

        return

    csv_files = sorted(INPUT_BASE.rglob("*.csv"))

    if not csv_files:

        print(f"No CSV files found in:\n" f"{INPUT_BASE}")

        return

    print(f"Model: {MODEL_NAME}")

    print(f"Perspective: " f"{PERSPECTIVE}")

    print(f"Found " f"{len(csv_files)} " f"OCEAN CSV files.\n")

    # ========================================================
    # PROCESS ALL CSV FILES
    # ========================================================

    all_rows = []

    for csv_path in csv_files:

        rows = process_csv(csv_path)

        all_rows.extend(rows)

    if not all_rows:

        print("No valid OCEAN " "distributions found.")

        return

    all_df = pd.DataFrame(all_rows)

    OUTPUT_BASE.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # PROCESS EACH GAME GENRE SEPARATELY
    # ========================================================

    for genre_name, safe_genre in GENRES.items():

        genre_df = all_df[all_df["game_genre"] == genre_name].copy()

        if genre_df.empty:

            print(f"\n No data found for " f"genre: {genre_name}")

            continue

        # Add percentage columns also
        # to the per-CSV output.

        for level in LEVELS:

            genre_df[f"{level}_pct"] = genre_df[level] * 100

        # ====================================================
        # MEAN DISTRIBUTION
        # ====================================================

        mean_df = compute_mean_distribution(genre_df)

        # ====================================================
        # SAVE PER-CSV DISTRIBUTIONS
        # ====================================================

        per_csv_path = OUTPUT_BASE / (
            f"{safe_genre}_" f"ocean_distributions_per_csv.csv"
        )

        genre_df.to_csv(
            per_csv_path,
            index=False,
        )

        # ====================================================
        # SAVE MEAN DISTRIBUTION
        # ====================================================

        mean_path = OUTPUT_BASE / (f"{safe_genre}_" f"ocean_mean_distribution.csv")

        mean_df.to_csv(
            mean_path,
            index=False,
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print("\n========================================")

        print(f"{genre_name.upper()}")

        print("========================================")

        for _, row in mean_df.iterrows():

            print(f"\n" f"{row['trait'].capitalize()}")

            print(f"  High:   " f"{row['high_pct']:.2f}%")

            print(f"  Medium: " f"{row['medium_pct']:.2f}%")

            print(f"  Low:    " f"{row['low_pct']:.2f}%")

            print(f"  CSVs:   " f"{int(row['n_csv'])}")

        print(f"\nSaved per-CSV data:\n" f"{per_csv_path}")

        print(f"\nSaved mean distribution:\n" f"{mean_path}")


if __name__ == "__main__":
    main()
