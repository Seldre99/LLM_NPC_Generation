import csv
from pathlib import Path
from collections import Counter, defaultdict

def analyze_error_csv(path: str):
    error_path = Path(path)

    if not error_path.exists():
        print("No error CSV found at:", error_path)
        return

    error_counter = Counter()
    genre_counter = Counter()
    race_counter = Counter()

    with open(error_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            error_counter[row["error"]] += 1
            genre_counter[row["genre"]] += 1
            race_counter[row["race"]] += 1

    print("\n=== Error Types ===")
    for err, count in error_counter.items():
        print(f"{err}: {count}")

    print("\n=== Errors per Genre ===")
    for g, count in genre_counter.items():
        print(f"{g}: {count}")

    print("\n=== Errors per Race ===")
    for r, count in race_counter.items():
        print(f"{r}: {count}")


if __name__ == "__main__":
    analyze_error_csv("errors/npc_quest/google_gemma-3-1b/english/errors_english.csv")
