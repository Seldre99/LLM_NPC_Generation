import re
import json
import csv
import lmstudio as lms
from pathlib import Path

SERVER_API_HOST = "localhost:1234"


def load_prompt_template(path: str) -> str:
    """Load the quest generation template."""
    template_path = Path(path)
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found at: {template_path}")
    return template_path.read_text(encoding="utf-8")

def build_system_text(template: str, genre: str, npc_race: str) -> str:
    """Replace placeholders in the template."""
    return (
        template
        .replace("[GENRE]", genre)
        .replace("[NPC_ETHNICITY]", npc_race)
    )

def build_user_text(template: str, role: str) -> str:
    return (
        template
        .replace("[ROLE_LIST]", str(role))
    )

def take_role(genre_index: int) -> dict:
    mapping = {
        1: "data/role_fantasy.json",
        2: "data/role_open_world.json",
        3: "data/role_cyberpunk.json",
        4: "data/role_crime.json"
    }
    if genre_index not in mapping:
        raise ValueError(f"No role mapping for genre index: {genre_index}")
    with open(mapping[genre_index], "r", encoding="utf-8") as f:
        return json.load(f)

def call_llm(system_text: str, user_text: str, used_model: str, temp: float) -> str:
    """Send the prompt to LM Studio. (VERSIONE BASE)
    with lms.Client() as client:
        model = client.llm.model(used_model)
        response = str(model.respond(input_text))

    # Remove <think> blocks
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    response = re.sub(r"```json\s*", "", response, flags=re.IGNORECASE)
    # Remove simple ```
    response = re.sub(r"```", "", response)
    return response
    """
    with lms.Client() as client:
        model = client.llm.model(used_model)
        chat = lms.Chat(system_text)
        chat.add_user_message(user_text)
        response = str(model.respond(chat, config={
            "temperature": temp
        }))

    # Remove <think> blocks
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    response = re.sub(r"```json\s*", "", response, flags=re.IGNORECASE)
    # Remove simple ```
    response = re.sub(r"```", "", response)
    return response


def validate_output(json_obj, expected_race, role_list):
    """Validazione JSON strutturata"""
    if not isinstance(json_obj, dict):
        return False, "not a JSON object"

    for field in ["npc name", "ethnicity", "role"]:
        if field not in json_obj:
            return False, f"missing field: {field}"

    if json_obj["ethnicity"].lower() != expected_race.lower():
        return False, "ethnicity mismatch"

    if json_obj["role"].lower() not in [r.lower() for r in role_list]:
        return False, "role not allowed"

    return True, None

def update_role_matrix_csv(csv_path: Path, roles_en: list, npc_details: list):
    """
    Maintain CSV with structure:
    ethnicity, role1, role2, ..., roleN
    Each cell is a cumulative count.
    roles_en: list of role names in English (column order).
    npc_details: list of {"name","ethnicity","role"} with ethnicity+role in English.
    """

    # Ensure CSV exists with header
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["ethnicity"] + roles_en
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Load existing table into dict
    csv_data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eth = row["ethnicity"]
            csv_data[eth] = {role: int(row.get(role, 0)) for role in roles_en}

    # Update counts
    for entry in npc_details:
        eth = entry["ethnicity"]
        role = entry["role"]
        if eth not in csv_data:
            csv_data[eth] = {r: 0 for r in roles_en}
        if role not in csv_data[eth]:
            # If the role is unknown (maybe mismatch), add column with 0 for existing rows
            csv_data[eth][role] = 0
            # Ensure all other rows have this role key present (keep CSV consistent)
            for other_eth in csv_data:
                if role not in csv_data[other_eth]:
                    csv_data[other_eth][role] = 0
            if role not in roles_en:
                roles_en.append(role)
        csv_data[eth][role] += 1

    # Write back CSV with columns ordered as ["ethnicity"] + roles_en
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["ethnicity"] + roles_en
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for eth, counts in csv_data.items():
            row = {"ethnicity": eth}
            # ensure all role columns exist
            for role in roles_en:
                row[role] = counts.get(role, 0)
            writer.writerow(row)


# -----------------------
# Update error matrix CSV per language
# -----------------------
def update_error_matrix_csv(csv_path: Path, error_entries: list):
    """
    Maintain per-language CSV:
    columns: genre, ethnicity, not a json object, missing field, ethnicity mismatch, role not allowed
    error_entries: list of {"genre":genre_en, "ethnicity":eth_en, "error":error_type}
    """
    error_types = ["not a json object", "missing field", "ethnicity mismatch", "role not allowed"]

    # Ensure CSV exists
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["genre", "ethnicity"] + error_types
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Load existing matrix
    matrix = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["genre"], row["ethnicity"])
            matrix[key] = {err: int(row.get(err, 0)) for err in error_types}

    # Normalize and update entries
    for e in error_entries:
        genre = e.get("genre", "")
        eth = e.get("ethnicity", "")
        raw_err = str(e.get("error", "")).strip().lower()

        if raw_err.startswith("missing field"):
            err_key = "missing field"
        elif raw_err in error_types:
            err_key = raw_err
        elif raw_err in ["invalid json", "not a json object"]:
            err_key = "not a json object"
        else:
            err_key = "not a json object"

        key = (genre, eth)
        if key not in matrix:
            matrix[key] = {err: 0 for err in error_types}
        matrix[key][err_key] += 1

    # Write back CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["genre", "ethnicity"] + error_types
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (genre, eth), counts in matrix.items():
            row = {"genre": genre, "ethnicity": eth}
            row.update(counts)
            writer.writerow(row)



def create_quest(sys_txt: str, user_txt: str, used_model: str, temp: float):
    lms.get_default_client(SERVER_API_HOST)

    with open("data/race_genre_language.json", "r", encoding="utf-8") as f:
        lang_data = json.load(f)

    languages = list(lang_data.keys())
    english_ethnicities = lang_data["english"]["npc_races"]
    english_genres = lang_data["english"]["genres"]

    for lang in languages:
        # load templates
        sys_template = load_prompt_template(f"prompt/{sys_txt}_{lang}.txt")
        usr_template = load_prompt_template(f"prompt/{user_txt}_{lang}.txt")

        print(f"\n=== Loading template for language: {lang} ===")

        genres = lang_data[lang]["genres"]
        npc_ethnicities = lang_data[lang]["npc_races"]

        # collect errors for this language-run (then update per-language csv)
        errors_to_update = []

        genre_type = 0

        for genre in genres:
            genre_type += 1

            # roles: local and english
            roles_data = take_role(genre_type)
            roles_local = roles_data[lang]["role"]
            roles_english = roles_data["english"]["role"]

            # aggregated json path for this genre/lang/model
            aggregated_dir = Path(f"output/npc_quest/{used_model.replace('/', '_')}/temperature_{temp}/{lang}")
            aggregated_dir.mkdir(parents=True, exist_ok=True)
            aggregated_filename = aggregated_dir / f"quests_{lang}_{english_genres[genre_type - 1]}.json"

            # load existing aggregated list or start empty
            if aggregated_filename.exists():
                with open(aggregated_filename, "r", encoding="utf-8") as f:
                    try:
                        all_quests_for_genre = json.load(f)
                        if not isinstance(all_quests_for_genre, list):
                            all_quests_for_genre = []
                    except:
                        all_quests_for_genre = []
            else:
                all_quests_for_genre = []

            npc_details = []

            for npc_ethnicity in npc_ethnicities:
                print(f"\n=== Generating {lang} quest for: {genre} + {npc_ethnicity} ===\n")

                system_text = build_system_text(sys_template, genre, npc_ethnicity)
                user_text = build_user_text(usr_template, roles_local)
                llm_response = call_llm(system_text, user_text, used_model, temp)

                ethnicity_index = npc_ethnicities.index(npc_ethnicity)
                ethnic_en = english_ethnicities[ethnicity_index]

                # parse JSON
                try:
                    json_obj = json.loads(llm_response)
                except Exception:
                    errors_to_update.append({
                        "genre": english_genres[genre_type - 1],
                        "ethnicity": ethnic_en,
                        "error": "not a json object"
                    })
                    continue

                # validate structure and content
                valid, reason = validate_output(json_obj, npc_ethnicity, roles_local)
                if not valid:
                    reason_norm = reason
                    if isinstance(reason, str) and reason.startswith("missing field"):
                        reason_norm = "missing field"
                    errors_to_update.append({
                        "genre": english_genres[genre_type - 1],
                        "ethnicity": ethnic_en,
                        "error": reason_norm
                    })
                    continue

                # map role to english
                try:
                    role_index = [r.lower() for r in roles_local].index(json_obj["role"].lower())
                    role_en = roles_english[role_index]
                except ValueError:
                    # role not found in mapping (shouldn't happen if validate passed), treat as role not allowed
                    errors_to_update.append({
                        "genre": english_genres[genre_type - 1],
                        "ethnicity": ethnic_en,
                        "error": "role not allowed"
                    })
                    continue

                print(llm_response)

                # append details for CSV matrix (ethnicity + role in English)
                npc_details.append({
                    "name": json_obj.get("npc name", ""),
                    "ethnicity": ethnic_en,
                    "role": role_en
                })

                # append full json to aggregated list
                all_quests_for_genre.append(json_obj)

            # save aggregated json (append mode)
            aggregated_filename.parent.mkdir(parents=True, exist_ok=True)
            with open(aggregated_filename, "w", encoding="utf-8") as f:
                json.dump(all_quests_for_genre, f, indent=4, ensure_ascii=False)

            # update role-matrix CSV for this genre/lang/model
            csv_dir = Path(f"csv/npc_quest/{used_model.replace('/', '_')}/temperature_{temp}/{lang}")
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_name = csv_dir / f"{english_genres[genre_type - 1]}_{used_model.replace('/', '_')}_{lang}.csv"
            update_role_matrix_csv(csv_name, roles_english.copy(), npc_details)

        # after processing all genres for this language, update per-language error CSV
        error_csv_path = Path(f"errors/{used_model.replace('/', '_')}/temperature_{temp}/{lang}/error_{lang}.csv")
        error_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if errors_to_update:
            update_error_matrix_csv(error_csv_path, errors_to_update)


# -----------------------
# Runner: N iterations
# -----------------------
if __name__ == "__main__":
    N = 1000  # number of total runs
    for i in range(N):
        print(f"\n======== ITERATION {i+1}/{N} ========\n")
        create_quest("npc_system", "npc_user", "google/gemma-3-1b", 0.7)
