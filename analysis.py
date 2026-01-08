import csv
from pathlib import Path


BASE_CSV_DIR = Path("csv/npc_quest")
OUTPUT_ANALYSIS_DIR = Path("analysis/npc_quest")

ANALYSIS_BASE = Path("analysis/npc_quest")
OUTPUT_BASE = Path("comparison/npc_quest")

TOP_N = 3

def analyze_role_distribution():
    """
    For each model / temperature / language / genre CSV:
    create per-ethnicity CSV sorted by role count desc.
    """

    for model_dir in BASE_CSV_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        for temp_dir in model_dir.iterdir():
            if not temp_dir.is_dir():
                continue

            for lang_dir in temp_dir.iterdir():
                if not lang_dir.is_dir():
                    continue

                for genre_csv in lang_dir.glob("*.csv"):
                    genre_name = genre_csv.stem.split("_")[0]

                    print(f"Analyzing → {genre_csv}")

                    with open(genre_csv, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        roles = reader.fieldnames[1:]  # exclude ethnicity

                        rows = list(reader)

                    # Output base dir
                    out_base = (
                        OUTPUT_ANALYSIS_DIR
                        / model_dir.name
                        / temp_dir.name
                        / lang_dir.name
                        / genre_name
                    )
                    out_base.mkdir(parents=True, exist_ok=True)

                    # For each ethnicity create sorted CSV
                    for row in rows:
                        ethnicity = row["ethnicity"]

                        role_counts = []
                        for role in roles:
                            try:
                                count = int(row.get(role, 0))
                            except ValueError:
                                count = 0
                            role_counts.append((role, count))

                        # sort descending
                        role_counts.sort(key=lambda x: x[1], reverse=True)

                        out_csv = out_base / f"{ethnicity}.csv"

                        with open(out_csv, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(["role", "count"])
                            for role, count in role_counts:
                                writer.writerow([role, count])

                        print(f"  ✓ {out_csv}")


def build_cross_language_comparison():
    for model_dir in ANALYSIS_BASE.iterdir():
        if not model_dir.is_dir():
            continue

        for temp_dir in model_dir.iterdir():
            if not temp_dir.is_dir():
                continue

            # Collect all languages
            languages = [
                d.name for d in temp_dir.iterdir() if d.is_dir()
            ]

            if not languages:
                continue

            # Assume genres are consistent across languages
            first_lang_dir = temp_dir / languages[0]
            genres = [d.name for d in first_lang_dir.iterdir() if d.is_dir()]

            for genre in genres:
                # Collect ethnicities from first language
                ethnicity_files = list((first_lang_dir / genre).glob("*.csv"))
                ethnicities = [f.stem for f in ethnicity_files]

                for ethnicity in ethnicities:
                    rows = []

                    for lang in languages:
                        ethnicity_csv = (
                            temp_dir / lang / genre / f"{ethnicity}.csv"
                        )

                        if not ethnicity_csv.exists():
                            continue

                        with open(ethnicity_csv, "r", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            top_roles = []

                            for i, row in enumerate(reader):
                                if i >= TOP_N:
                                    break
                                top_roles.append((row["role"], row["count"]))

                        row_data = {
                            "language": lang
                        }

                        for i in range(TOP_N):
                            if i < len(top_roles):
                                role, count = top_roles[i]
                                row_data[f"role_{i+1}"] = role
                                row_data[f"count_{i+1}"] = count
                            else:
                                row_data[f"role_{i+1}"] = ""
                                row_data[f"count_{i+1}"] = 0

                        rows.append(row_data)

                    # Write output CSV
                    out_dir = (
                        OUTPUT_BASE
                        / model_dir.name
                        / temp_dir.name
                        / genre
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)

                    out_csv = out_dir / f"{ethnicity}.csv"

                    fieldnames = (
                        ["language"]
                        + [f"role_{i+1}" for i in range(TOP_N)]
                        + [f"count_{i+1}" for i in range(TOP_N)]
                    )

                    with open(out_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)

                    print(f"✓ Created comparison: {out_csv}")

if __name__ == "__main__":
    #analyze_role_distribution()
    build_cross_language_comparison()
