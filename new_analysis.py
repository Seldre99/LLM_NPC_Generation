import csv
from pathlib import Path
import json

ANALYSIS_BASE = Path("analysis/npc_quest")
ERROR_BASE = Path("errors")
OUTPUT_BASE = Path("comparison/npc_quest")

TOP_N = 3


def load_error_counts(error_csv, genre, ethnicity):
    counts = {
        "not a json object": 0,
        "missing field": 0,
        "ethnicity mismatch": 0,
        "role not allowed": 0,
    }

    if not error_csv.exists():
        return counts

    with open(error_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["genre"] == genre and row["ethnicity"] == ethnicity:
                for k in counts:
                    counts[k] += int(row.get(k, 0))
    return counts


def build_percentage_and_error_analysis():
    for model_dir in ANALYSIS_BASE.iterdir():
        if not model_dir.is_dir():
            continue

        for temp_dir in model_dir.iterdir():
            if not temp_dir.is_dir():
                continue

            languages = [d.name for d in temp_dir.iterdir() if d.is_dir()]
            if not languages:
                continue

            first_lang_dir = temp_dir / languages[0]
            genres = [d.name for d in first_lang_dir.iterdir() if d.is_dir()]

            for genre in genres:
                ethnicities = [
                    f.stem for f in (first_lang_dir / genre).glob("*.csv")
                ]

                for ethnicity in ethnicities:
                    role_rows = []
                    error_rows = []

                    for lang in languages:
                        role_csv = (
                            temp_dir / lang / genre / f"{ethnicity}.csv"
                        )

                        if not role_csv.exists():
                            continue

                        # --- ROLE COUNTS ---
                        role_counts = []
                        total_valid = 0

                        with open(role_csv, "r", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                count = int(row["count"])
                                total_valid += count
                                role_counts.append((row["role"], count))

                        # --- ERRORS ---
                        error_csv = (
                            ERROR_BASE
                            / model_dir.name
                            / temp_dir.name
                            / lang
                            / f"error_{lang}.csv"
                        )

                        error_counts = load_error_counts(
                            error_csv, genre, ethnicity
                        )
                        total_errors = sum(error_counts.values())

                        error_counts["language"] = lang
                        error_counts["total_errors"] = total_errors
                        error_rows.append(error_counts)

                        # --- PERCENTAGES ---
                        denominator = total_valid + total_errors
                        top_roles = role_counts[:TOP_N]

                        row = {"language": lang}

                        for i in range(TOP_N):
                            if i < len(top_roles) and denominator > 0:
                                role, count = top_roles[i]
                                pct = round((count / denominator) * 100, 2)
                                row[f"role_{i+1}"] = role
                                row[f"pct_{i+1}"] = pct
                            else:
                                row[f"role_{i+1}"] = ""
                                row[f"pct_{i+1}"] = 0.0

                        role_rows.append(row)

                    # OUTPUT DIR
                    out_dir = (
                        OUTPUT_BASE
                        / model_dir.name
                        / temp_dir.name
                        / genre
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # --- WRITE ROLE % CSV ---
                    role_csv_out = out_dir / f"{ethnicity}_roles_percent.csv"
                    role_fields = (
                        ["language"]
                        + [f"role_{i+1}" for i in range(TOP_N)]
                        + [f"pct_{i+1}" for i in range(TOP_N)]
                    )

                    with open(role_csv_out, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=role_fields)
                        writer.writeheader()
                        writer.writerows(role_rows)

                    # --- WRITE ERROR CSV ---
                    error_csv_out = out_dir / f"{ethnicity}_errors.csv"
                    error_fields = [
                        "language",
                        "not a json object",
                        "missing field",
                        "ethnicity mismatch",
                        "role not allowed",
                        "total_errors",
                    ]

                    with open(error_csv_out, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=error_fields)
                        writer.writeheader()
                        for e in error_rows:
                            writer.writerow(e)

                    print(f"✓ {genre} | {ethnicity} analysis done")

def normalize(text: str) -> str:
    return text.lower().replace(" ", "_")

def build_first_choice_by_language():
    base_dir = Path("comparison/npc_quest")

    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue

        for temp_dir in model_dir.iterdir():
            if not temp_dir.is_dir():
                continue

            for genre_dir in temp_dir.iterdir():
                if not genre_dir.is_dir():
                    continue

                role_files = list(genre_dir.glob("*_roles_percent.csv"))
                if not role_files:
                    continue

                # lingua -> rows
                per_language = {}

                for role_file in role_files:
                    ethnicity = role_file.stem.replace("_roles_percent", "")

                    with open(role_file, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            lang = row["language"]

                            if not row["role_1"]:
                                continue

                            per_language.setdefault(lang, []).append({
                                "ethnicity": ethnicity,
                                "role": row["role_1"],
                                "percentage": row["pct_1"]
                            })

                # OUTPUT DIR
                out_dir = genre_dir / "first_choice_by_language"
                out_dir.mkdir(parents=True, exist_ok=True)

                # WRITE CSV PER LINGUA
                for lang, rows in per_language.items():
                    out_csv = out_dir / f"{lang}.csv"

                    with open(out_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=["ethnicity", "role", "percentage"]
                        )
                        writer.writeheader()
                        writer.writerows(rows)

                print(
                    f"✓ First-choice CSVs created for genre: {genre_dir.name}"
                )

def build_role_coverage_csv():
    base_csv_dir = Path("csv/npc_quest")
    role_json_dir = Path("data")

    genre_role_map = {
        "Fantasy RPG": "role_fantasy.json",
        "Contemporary Open World": "role_open_world.json",
        "Cyberpunk": "role_cyberpunk.json",
        "Crime": "role_crime.json",
    }

    # Normalized lookup
    normalized_genre_map = {
        normalize(k): v for k, v in genre_role_map.items()
    }

    for model_dir in base_csv_dir.iterdir():
        if not model_dir.is_dir():
            continue

        for temp_dir in model_dir.iterdir():
            if not temp_dir.is_dir():
                continue

            for lang_dir in temp_dir.iterdir():
                if not lang_dir.is_dir():
                    continue

                language = lang_dir.name

                for csv_file in lang_dir.glob("*.csv"):

                    filename_norm = normalize(csv_file.stem)

                    genre_key = None
                    for g_norm in normalized_genre_map:
                        if filename_norm.startswith(g_norm):
                            genre_key = g_norm
                            break

                    if genre_key is None:
                        continue

                    genre = genre_key.replace("_", " ").title()

                    # LOAD ROLE JSON
                    role_json_path = role_json_dir / normalized_genre_map[genre_key]
                    with open(role_json_path, "r", encoding="utf-8") as f:
                        role_data = json.load(f)

                    total_roles = len(role_data["english"]["role"])

                    # READ CSV
                    with open(csv_file, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)

                        for row in reader:
                            ethnicity = row["ethnicity"]

                            roles_used = 0
                            for role, value in row.items():
                                if role in ["ethnicity", "name"]:
                                    continue
                                try:
                                    if int(value) > 0:
                                        roles_used += 1
                                except:
                                    pass

                            # OUTPUT
                            out_dir = Path(
                                f"analysis/role_coverage/"
                                f"{model_dir.name}/{temp_dir.name}/{genre.replace(' ', '_')}"
                            )
                            out_dir.mkdir(parents=True, exist_ok=True)

                            out_csv = out_dir / f"{ethnicity}_roles_coverage.csv"

                            file_exists = out_csv.exists()

                            with open(out_csv, "a", newline="", encoding="utf-8") as f:
                                writer = csv.DictWriter(
                                    f,
                                    fieldnames=["language", "roles_used", "total_roles"]
                                )
                                if not file_exists:
                                    writer.writeheader()

                                writer.writerow({
                                    "language": language,
                                    "roles_used": roles_used,
                                    "total_roles": total_roles
                                })

                            print(
                                f"✓ {genre} | {ethnicity} | {language} → "
                                f"{roles_used}/{total_roles}"
                            )





if __name__ == "__main__":
    #build_percentage_and_error_analysis()
    #build_first_choice_by_language()
    build_role_coverage_csv()
