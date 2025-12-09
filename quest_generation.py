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

def take_role(genre: int) -> dict:
    if genre == 1:
        json_path = "data/role_fantasy.json"
    elif genre == 2:
        json_path = "data/role_open_world.json"
    elif genre == 3:
        json_path = "data/role_cyberpunk.json"
    elif genre == 4:
        json_path = "data/role_crime.json"

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def call_llm(system_text: str, user_text: str, used_model: str) -> str:
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
        print(user_text)
        response = str(model.respond(chat))

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

def create_quest(sys_txt: str, user_txt: str, used_model: str):
    """Put together and create the quest."""
    lms.get_default_client(SERVER_API_HOST)

    # JSON con generi e razze tradotti
    json_path = "data/race_genre_language.json"

    with open(json_path, "r", encoding="utf-8") as f:
        lang_data = json.load(f)

    # Lista lingue da processare
    languages = list(lang_data.keys())

    english_races = lang_data["english"]["npc_races"]
    english_genres = lang_data["english"]["genres"]

    error_count = 0

    for lang in languages:
        genre_type = 0
        #template_path = f"prompt/quest_generation_{lang}.txt"
        sys_template_path = f"prompt/{sys_txt}_{lang}.txt"
        usr_template_path = f"prompt/{user_txt}_{lang}.txt"
        print(f"\n=== Loading template for language: {lang} ===")

        sys_template = load_prompt_template(sys_template_path)
        usr_template = load_prompt_template(usr_template_path)

        # Dati localizzati da JSON
        genres = lang_data[lang]["genres"]
        npc_races = lang_data[lang]["npc_races"]

        errors_for_lang = []

        for genre in genres:

            genre_type += 1
            npc_details = []
            all_quests_for_genre = []

            for npc_race in npc_races:
                print(f"\n=== Generating {lang} quest for: {genre} + {npc_race} ===\n")

                roles_data = take_role(genre_type)
                roles = roles_data[lang]["role"]
                print(roles)

                system_text = build_system_text(sys_template, genre, npc_race)
                user_text = build_user_text(usr_template, roles)
                llm_response = call_llm(system_text, user_text, used_model)

                race_index = npc_races.index(npc_race)
                race_en = english_races[race_index]


                # 1. Validate JSON
                try:
                    json_obj = json.loads(llm_response)
                except:
                    print("Invalid JSON — skipped")

                    errors_for_lang.append({
                        "genre": english_genres[genre_type - 1],
                        "race": english_races[race_index],
                        "error": "invalid json"
                    })
                    continue

                # 2. Structured validation
                valid, reason = validate_output(json_obj, npc_race, roles)
                if not valid:
                    print(f"✖ Invalid output ({reason}) — skipped")
                    errors_for_lang.append({
                        "genre": english_genres[genre_type - 1],
                        "race": english_races[race_index],
                        "error": reason
                    })
                    continue

                print(f"LLM response: {llm_response}")

                # Output folder: /output/<lang>/
                #out_dir = Path(f"output/{used_model.replace('/', '_')}/{lang}")
                #out_dir.mkdir(parents=True, exist_ok=True)

                #filename = out_dir / f"quest_{lang}_{genre}_{npc_race}.json"
                #filename.write_text(llm_response, encoding="utf-8")

                role_index = [r.lower() for r in roles].index(json_obj["role"].lower())
                role_en = roles_data["english"]["role"][role_index]

                # 4. Prepare row for CSV
                npc_details.append({
                    "name": json_obj["npc name"],
                    "race": race_en,
                    "role": role_en
                })

                all_quests_for_genre.append(json_obj)

            # 6. CREATE AGGREGATED JSON FOR GENRE (NEW)
            aggregated_json_dir = Path(f"output/npc_quest/{used_model.replace('/', '_')}/{lang}")
            aggregated_json_dir.mkdir(parents=True, exist_ok=True)

            aggregated_filename = aggregated_json_dir / f"quests_{lang}_{genre}.json"

            with open(aggregated_filename, "w", encoding="utf-8") as f:
                json.dump(all_quests_for_genre, f, indent=4, ensure_ascii=False)

            # 5. CREATE CSV FOR GENRE
            csv_dir = Path(f"csv/npc_quest/{used_model.replace('/', '_')}/{lang}")
            csv_dir.mkdir(parents=True, exist_ok=True)

            csv_name = csv_dir / f"{genre}_{used_model.replace('/', '_')}_{lang}.csv"

            with open(csv_name, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["name", "race", "role"])
                writer.writeheader()
                writer.writerows(npc_details)

            print(f"✓ CSV created: {csv_name}")

        # Save error CSV
        if errors_for_lang:
            error_dir = Path(f"errors/npc_quest/{used_model.replace('/', '_')}/{lang}")
            error_dir.mkdir(parents=True, exist_ok=True)

            error_csv = error_dir / f"errors_{lang}.csv"

            with open(error_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["genre", "race", "error"])
                writer.writeheader()
                writer.writerows(errors_for_lang)

            print(f"⚠ Error CSV created: {error_csv}")

    print(f"\n=== Completed with {error_count} invalid responses ===")


if __name__ == "__main__":
    create_quest("npc_system", "npc_user", "google/gemma-3-1b")
    #create_quest("npc_quest", "qwen/qwen3-8b")