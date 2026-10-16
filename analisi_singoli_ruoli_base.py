import os
import glob
import pandas as pd

NPC_BASE    = "analysis/npc_quest/qwen"
WRITER_BASE = "analysis/writer/qwen"
OUTPUT_BASE = "analysis/roles/npc_vs_writer/qwen"

MODEL_NAME      = "Qwen_Qwen3-8B"
LABEL_NPC       = "Qwen3-8B-npc"
LABEL_WRITER    = "Qwen3-8B-writer"


def parse_csv_path(path: str, base_marker: str) -> dict | None:
    parts = os.path.normpath(path).split(os.sep)
    try:
        base_idx = parts.index(base_marker)
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


def process_csv(path: str, base_marker: str, label: str) -> None:
    meta = parse_csv_path(path, base_marker)
    if meta is None:
        return

    # Processa solo il modello di interesse
    if meta["modello"] != MODEL_NAME:
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  Impossibile leggere {path}: {e}")
        return

    if df.empty or "role" not in df.columns or "count" not in df.columns:
        print(f"  Struttura non valida, skip: {path}")
        return

    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

    total = df["count"].sum()
    if total == 0:
        print(f" Tutti i conteggi sono 0, skip: {path}")
        return

    for _, row in df.iterrows():
        role  = str(row["role"]).strip()
        count = int(row["count"])
        other = total - count

        out_path = build_role_output_path(meta, role)
        new_row  = pd.DataFrame([{
            "modello": label,
            "count":   count,
            "other":   other,
        }])

        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            existing = existing[existing["modello"] != label]
            updated  = pd.concat([existing, new_row], ignore_index=True)
        else:
            updated = new_row

        updated.to_csv(out_path, index=False)

    print(f" [{label}] {meta['genere_vg']} | {meta['genere_npc']} | {meta['body_type']} | {meta['etnia']}")


def main() -> None:
    # ── File da npc_quest ──────────────────────────────────────────────────
    npc_files = glob.glob(
        os.path.join(NPC_BASE, MODEL_NAME, "**", "*.csv"), recursive=True
    )
    print(f"Trovati {len(npc_files)} file CSV in npc_quest.\n")
    for path in sorted(npc_files):
        process_csv(path, "npc_quest", LABEL_NPC)

    # ── File da writer ─────────────────────────────────────────────────────
    writer_files = glob.glob(
        os.path.join(WRITER_BASE, MODEL_NAME, "**", "*.csv"), recursive=True
    )
    print(f"\nTrovati {len(writer_files)} file CSV in writer.\n")
    for path in sorted(writer_files):
        process_csv(path, "writer", LABEL_WRITER)

    print("\nCompletato.")


if __name__ == "__main__":
    main()
