import os
import glob
import pandas as pd

INPUT_BASE  = "analysis/writer/qwen"
OUTPUT_BASE = "analysis/roles/writer/qwen"

PATTERN = os.path.join(INPUT_BASE, "**", "*.csv")


def parse_csv_path(path: str) -> dict | None:
    parts = os.path.normpath(path).split(os.sep)
    try:
        base_idx = parts.index("writer")
    except ValueError:
        return None

    rel = parts[base_idx + 2:]
    if len(rel) < 7:
        return None

    return {
        "modello":    rel[0],
        "temp":       rel[1],
        "genere_vg":  rel[3],
        "genere_npc": rel[4],
        "body_type":  rel[5],
        "etnia":      os.path.splitext(rel[6])[0],
    }


def build_role_output_path(meta: dict, role: str) -> str:
    safe_role = role.replace(" ", "_").replace("/", "-")
    out_dir = os.path.join(
        OUTPUT_BASE,
        meta["genere_vg"],
        meta["genere_npc"],
        meta["body_type"],
        meta["etnia"],
    )
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{safe_role}.csv")


def process_csv(path: str) -> None:
    meta = parse_csv_path(path)
    if meta is None:
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f" Impossibile leggere {path}: {e}")
        return

    if df.empty or "role" not in df.columns or "count" not in df.columns:
        print(f" Struttura non valida, skip: {path}")
        return

    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

    total = df["count"].sum()
    if total == 0:
        print(f" Tutti i conteggi sono 0, skip: {path}")
        return

    modello_label = meta["modello"]

    for _, row in df.iterrows():
        role      = str(row["role"]).strip()
        count     = int(row["count"])
        other     = total - count

        out_path  = build_role_output_path(meta, role)
        new_row   = pd.DataFrame([{
            "modello": modello_label,
            "count":   count,
            "other":   other,
        }])

        # Aggiorna il CSV se esiste, altrimenti lo crea
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            # Evita duplicati: rimuovi eventuale riga con stesso modello
            existing = existing[existing["modello"] != modello_label]
            updated  = pd.concat([existing, new_row], ignore_index=True)
        else:
            updated = new_row

        updated.to_csv(out_path, index=False)

    print(f"{meta['modello']} | {meta['genere_vg']} | {meta['genere_npc']} | {meta['body_type']} | {meta['etnia']}")


def main() -> None:
    csv_files = glob.glob(PATTERN, recursive=True)

    if not csv_files:
        print(f"Nessun CSV trovato: {PATTERN}")
        return

    print(f"Trovati {len(csv_files)} file CSV.\n")

    for path in sorted(csv_files):
        process_csv(path)

    print("\n Completato.")


if __name__ == "__main__":
    main()
