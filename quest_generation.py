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

def build_system_text(template: str, genre: str, npc_race: str, npc_genre: str, npc_body: str) -> str:
    """Replace placeholders in the template."""
    return (
        template
        .replace("[GENRE]", genre)
        .replace("[NPC_ETHNICITY]", npc_race)
        .replace("[NPC_GENRE]", npc_genre)
        .replace("[NPC_BODY]", npc_body)
    )

def build_user_text(template: str, role: str) -> str:
    return (
        template
        .replace("[ROLE_LIST]", str(role))
    )

def take_role(genre_index: int) -> dict:
    mapping = {
        1: "data/role_fantasy.json",
        2: "data/role_open_world.json"
        #3: "data/role_cyberpunk.json",
        #4: "data/role_crime.json"
    }
    if genre_index not in mapping:
        raise ValueError(f"No role mapping for genre index: {genre_index}")
    with open(mapping[genre_index], "r", encoding="utf-8") as f:
        return json.load(f)

def call_llm(system_text: str, user_text: str, used_model: str, temp: float) -> str:
    with lms.Client() as client:
        model = client.llm.model(used_model)
        chat = lms.Chat(system_text)
        chat.add_user_message(user_text)
        response = str(model.respond(chat, config={
            "temperature": temp
        }))

    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    response = re.sub(r"```json\s*", "", response, flags=re.IGNORECASE)
    response = re.sub(r"```", "", response)
    return response


def validate_output(role: str, role_list: list) -> tuple:
    """Controlla se il ruolo restituito è presente nella lista dei ruoli consentiti."""
    if role.strip().lower() not in [r.lower() for r in role_list]:
        return False, "role not allowed"
    return True, None


def update_role_matrix_csv(csv_path: Path, roles_en: list, ethnicity: str, npc_role: str):
    """
    Maintain CSV with structure:
    ethnicity, role1, role2, ..., roleN
    Each cell is a cumulative count.
    ethnicity: etnicity string in English.
    npc_role: role string in English.
    """

    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["ethnicity"] + roles_en
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    csv_data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eth = row["ethnicity"]
            csv_data[eth] = {role: int(row.get(role, 0)) for role in roles_en}

    if ethnicity not in csv_data:
        csv_data[ethnicity] = {r: 0 for r in roles_en}
    if npc_role not in csv_data[ethnicity]:
        csv_data[ethnicity][npc_role] = 0
        for other_eth in csv_data:
            if npc_role not in csv_data[other_eth]:
                csv_data[other_eth][npc_role] = 0
        if npc_role not in roles_en:
            roles_en.append(npc_role)
    csv_data[ethnicity][npc_role] += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["ethnicity"] + roles_en
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for eth, counts in csv_data.items():
            row = {"ethnicity": eth}
            for role in roles_en:
                row[role] = counts.get(role, 0)
            writer.writerow(row)


def update_error_matrix_csv(csv_path: Path, error_entries: list):
    """
    Maintain per-language error CSV.
    columns: genre, ethnicity, npc_genre, npc_body, role not allowed
    error_entries: list of {"genre", "ethnicity", "npc_genre", "npc_body", "error"}
    """
    error_types = ["role not allowed"]

    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["genre", "ethnicity", "npc_genre", "npc_body"] + error_types
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    matrix = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["genre"], row["ethnicity"], row["npc_genre"], row["npc_body"])
            matrix[key] = {err: int(row.get(err, 0)) for err in error_types}

    for e in error_entries:
        genre    = e.get("genre", "")
        eth      = e.get("ethnicity", "")
        npc_genre = e.get("npc_genre", "")
        npc_body  = e.get("npc_body", "")

        key = (genre, eth, npc_genre, npc_body)
        if key not in matrix:
            matrix[key] = {err: 0 for err in error_types}
        matrix[key]["role not allowed"] += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["genre", "ethnicity", "npc_genre", "npc_body"] + error_types
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (genre, eth, npc_genre, npc_body), counts in matrix.items():
            row = {"genre": genre, "ethnicity": eth, "npc_genre": npc_genre, "npc_body": npc_body}
            row.update(counts)
            writer.writerow(row)


def create_npc(sys_txt: str, user_txt: str, used_model: str, temp: float):

    with open("data/race_genre_language.json", "r", encoding="utf-8") as f:
        lang_data = json.load(f)

    languages          = list(lang_data.keys())
    english_ethnicities = lang_data["english"]["npc_races"]
    english_genres      = lang_data["english"]["genres"]
    english_npc_genres  = lang_data["english"]["npc_genres"]
    english_npc_bodies  = lang_data["english"]["npc_bodies"]

    for lang in languages:
        sys_template = load_prompt_template(f"prompt/{sys_txt}_{lang}.txt")
        usr_template = load_prompt_template(f"prompt/{user_txt}_{lang}.txt")

        print(f"\n=== Loading template for language: {lang} ===")

        genres         = lang_data[lang]["genres"]
        npc_ethnicities = lang_data[lang]["npc_races"]
        npc_genres      = lang_data[lang]["npc_genres"]
        npc_bodies      = lang_data[lang]["npc_bodies"]

        errors_to_update = []
        genre_type = 0

        for genre in genres:
            genre_type += 1

            roles_data   = take_role(genre_type)
            roles_local   = roles_data[lang]["role"]
            roles_english = roles_data["english"]["role"]

            for npc_ethnicity in npc_ethnicities:
                ethnicity_index = npc_ethnicities.index(npc_ethnicity)
                ethnic_en       = english_ethnicities[ethnicity_index]

                for npc_genre in npc_genres:
                    npc_genre_index = npc_genres.index(npc_genre)
                    npc_genre_en    = english_npc_genres[npc_genre_index]

                    for npc_body in npc_bodies:
                        npc_body_index = npc_bodies.index(npc_body)
                        npc_body_en    = english_npc_bodies[npc_body_index]

                        print(
                            f"\n=== Generating {lang} NPC: "
                            f"{genre} | {npc_ethnicity} | {npc_genre} | {npc_body} ===\n"
                        )

                        system_text   = build_system_text(sys_template, genre, npc_ethnicity, npc_genre, npc_body)
                        user_text     = build_user_text(usr_template, roles_local)
                        role_response = call_llm(system_text, user_text, used_model, temp).strip()

                        # validate role string
                        valid, reason = validate_output(role_response, roles_local)
                        if not valid:
                            errors_to_update.append({
                                "genre":     english_genres[genre_type - 1],
                                "ethnicity": ethnic_en,
                                "npc_genre": npc_genre_en,
                                "npc_body":  npc_body_en,
                                "error":     reason
                            })
                            continue

                        # map role to English
                        try:
                            role_index = [r.lower() for r in roles_local].index(role_response.lower())
                            role_en    = roles_english[role_index]
                        except ValueError:
                            errors_to_update.append({
                                "genre":     english_genres[genre_type - 1],
                                "ethnicity": ethnic_en,
                                "npc_genre": npc_genre_en,
                                "npc_body":  npc_body_en,
                                "error":     "role not allowed"
                            })
                            continue

                        print(role_response)

                        # update role-matrix CSV per genre / lang / npc_genre / npc_body / model
                        csv_dir = Path(
                            f"csv/npc/{used_model.replace('/', '_')}/temperature_{temp}"
                            f"/{lang}/{english_genres[genre_type - 1]}/{npc_genre_en}/{npc_body_en}"
                        )
                        csv_dir.mkdir(parents=True, exist_ok=True)
                        csv_name = csv_dir / (
                            f"{english_genres[genre_type - 1]}_{npc_genre_en}_{npc_body_en}"
                            f"_{used_model.replace('/', '_')}_{lang}.csv"
                        )
                        update_role_matrix_csv(csv_name, roles_english.copy(), ethnic_en, role_en)

        # after all genres for this language, update the per-language error CSV
        error_csv_path = Path(
            f"errors/{used_model.replace('/', '_')}/temperature_{temp}/{lang}/error_{lang}.csv"
        )
        error_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if errors_to_update:
            update_error_matrix_csv(error_csv_path, errors_to_update)


# -----------------------
# Runner: N iterations
# -----------------------
if __name__ == "__main__":
    N = 1000
    lms.get_default_client(SERVER_API_HOST)
    for i in range(N):
        print(f"\n======== ITERATION {i+1}/{N} ========\n")
        create_npc("npc_system", "npc_user", "google/gemma-3-1b", 0.7)
