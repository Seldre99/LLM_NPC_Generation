import re
import json
import csv
#import lmstudio as lms
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from huggingface_hub import login

login(token="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")

#SERVER_API_HOST = "localhost:1234"


# Carica modello una volta sola fuori da create_npc, non ad ogni chiamata
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.eval()
    tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role']+'<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %} {% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    return tokenizer, model, device

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

def build_user_text(template: str, role: str, npc_race: str, npc_genre: str, npc_body: str) -> str:
    return (
        template
        .replace("[ROLE_LIST]", str(role))
        #.replace("[NPC_ETHNICITY]", npc_race)
        #.replace("[NPC_GENRE]", npc_genre)
        #.replace("[NPC_BODY]", npc_body)
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

def call_llm(system_text: str, user_text: str, tokenizer, model, temp: float, device) -> str:
    #prompt = f"{system_text}\n\n{user_text}\n"

    conversation = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text}
    ]

    inputs = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt").to(device)


    #inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            gen_tokens = model.generate(
                inputs,
                max_new_tokens=150,
                temperature=temp,
                stop_strings=["<|eot_id|>"],
                tokenizer=tokenizer,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
            #output = model.generate(
            #    **inputs,
            #    max_new_tokens=80,
            #    temperature=temp,
            #    do_sample=True,
            #    pad_token_id=tokenizer.eos_token_id
            #)


        # Decodifica solo i token generati, non il prompt
        #generated = output[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(gen_tokens[0][len(inputs[0]): -1])
        #response = tokenizer.decode(generated, skip_special_tokens=True).strip()

        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        response = re.sub(r"```json\s*", "", response, flags=re.IGNORECASE)
        response = re.sub(r"```", "", response)
        response = re.search(r'\{.*\}', response, re.DOTALL)
        response = response.group(0)
        return response

    except Exception as e:
        return ""


def validate_output(raw_response: str, role_list: list) -> tuple:
    """Controlla se l'output è un JSON, il ruolo restituito è presente nella lista dei ruoli consentiti e c'è il campo personality con OCEAN."""
    OCEAN_FIELDS = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
    OCEAN_VALUES = {"high", "medium", "low"}

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        return False, "invalid json", None

    if not isinstance(data, dict):
        return False, "invalid json", None

    if "role" not in data:
        return False, "invalid json", None

    if "personality" not in data:
        return False, "missing personality", None

    role_value = data["role"]
    if not isinstance(role_value, str):
        return False, "invalid json", None

    if role_value.strip().lower() not in [r.lower() for r in role_list]:
        return False, "role not allowed", None

    personality = data["personality"]
    if not isinstance(personality, dict):
        return False, "missing personality", None

    # Controlla che ci siano tutti e 5 i campi OCEAN con valori validi
    if set(personality.keys()) != OCEAN_FIELDS:
        return False, "missing personality", None

    for field, value in personality.items():
        if not isinstance(value, str) or value.strip().lower() not in OCEAN_VALUES:
            return False, "missing personality", None

    return True, None, role_value.strip().lower()


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
    error_types = ["role not allowed", "invalid json", "missing personality"]

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
        genre = e.get("genre", "")
        eth = e.get("ethnicity", "")
        npc_genre = e.get("npc_genre", "")
        npc_body = e.get("npc_body", "")
        error = e.get("error", "")

        key = (genre, eth, npc_genre, npc_body)
        if key not in matrix:
            matrix[key] = {err: 0 for err in error_types}

        if error in error_types:
            matrix[key][error] += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["genre", "ethnicity", "npc_genre", "npc_body"] + error_types
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (genre, eth, npc_genre, npc_body), counts in matrix.items():
            row = {"genre": genre, "ethnicity": eth, "npc_genre": npc_genre, "npc_body": npc_body}
            row.update(counts)
            writer.writerow(row)


def save_npc_output(output_dir: Path, used_model: str,temp: float, lang: str, genre_en: str, npc_genre_en: str, npc_body_en: str, ethnic_en: str, raw_response: str):
    """
    Salva il JSON dell'LLM in un file dedicato per ogni combinazione NPC. Il file accumula tutti gli output generati per quella combinazione.
    """
    save_dir = Path(
        f"output/npc/{used_model.replace('/', '_')}/temperature_{temp}"
        f"/{lang}/{genre_en}/{npc_genre_en}/{npc_body_en}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    file_name = save_dir / f"{genre_en}_{npc_genre_en}_{npc_body_en}_{ethnic_en}_{lang}.json"

    # Carica lista esistente o inizia da zero
    existing = []
    if file_name.exists():
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing = []

    # Aggiunge il nuovo output
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        parsed = {"raw": raw_response}  # salva comunque in caso di JSON malformato

    existing.append(parsed)

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def create_npc(sys_txt: str, user_txt: str, used_model: str, temp: float, iteration: int, tokenizer, model, pbar, device):

    with open("data/race_genre_language.json", "r", encoding="utf-8") as f:
        lang_data = json.load(f)

    lang = "english"

    english_ethnicities = lang_data["english"]["npc_races"]
    english_genres      = lang_data["english"]["genres"]
    english_npc_genres  = lang_data["english"]["npc_genres"]
    english_npc_bodies  = lang_data["english"]["npc_bodies"]

    sys_template = load_prompt_template(f"prompt/{sys_txt}_{lang}.txt")
    usr_template = load_prompt_template(f"prompt/{user_txt}_{lang}.txt")

    genres        = lang_data[lang]["genres"]
    npc_ethnicities = lang_data[lang]["npc_races"]
    npc_genres    = lang_data[lang]["npc_genres"]
    npc_bodies    = lang_data[lang]["npc_bodies"]

    genre_type = 0

    for genre in genres:
        genre_type += 1

        roles_data    = take_role(genre_type)
        roles_local   = roles_data[lang]["role"]
        roles_english = roles_data["english"]["role"]

        errors_to_update = []

        for npc_ethnicity in npc_ethnicities:
            ethnicity_index = npc_ethnicities.index(npc_ethnicity)
            ethnic_en       = english_ethnicities[ethnicity_index]

            for npc_genre in npc_genres:
                npc_genre_index = npc_genres.index(npc_genre)
                npc_genre_en    = english_npc_genres[npc_genre_index]

                for npc_body in npc_bodies:
                    npc_body_index = npc_bodies.index(npc_body)
                    npc_body_en    = english_npc_bodies[npc_body_index]

                    # Aggiorna descrizione barra
                    pbar.set_description(
                        f"Iter {iteration} | {genre} | {npc_ethnicity} | {npc_genre} | {npc_body}"
                    )

                    system_text   = build_system_text(sys_template, genre, npc_ethnicity, npc_genre, npc_body)
                    user_text     = build_user_text(usr_template, roles_local, npc_ethnicity, npc_genre, npc_body)
                    role_response = call_llm(system_text, user_text, tokenizer, model, temp, device).strip()

                    #print(role_response)


                    valid, reason, role_taken = validate_output(role_response, roles_local)
                    if not valid:
                        errors_to_update.append({
                            "genre":     english_genres[genre_type - 1],
                            "ethnicity": ethnic_en,
                            "npc_genre": npc_genre_en,
                            "npc_body":  npc_body_en,
                            "error":     reason
                        })
                        pbar.update(1)
                        continue

                    save_npc_output(
                        output_dir=Path("output"),
                        used_model=used_model,
                        temp=temp,
                        lang=lang,
                        genre_en=english_genres[genre_type - 1],
                        npc_genre_en=npc_genre_en,
                        npc_body_en=npc_body_en,
                        ethnic_en=ethnic_en,
                        raw_response=role_response
                    )

                    try:
                        role_index = [r.lower() for r in roles_local].index(role_taken.lower())
                        role_en    = roles_english[role_index]
                    except ValueError:
                        errors_to_update.append({
                            "genre":     english_genres[genre_type - 1],
                            "ethnicity": ethnic_en,
                            "npc_genre": npc_genre_en,
                            "npc_body":  npc_body_en,
                            "error":     "role not allowed"
                        })
                        pbar.update(1)
                        continue

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
                    pbar.update(1)

        error_csv_path = Path(
            f"errors/npc/{used_model.replace('/', '_')}/temperature_{temp}/{lang}/error_{lang}.csv"
        )
        error_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if errors_to_update:
            update_error_matrix_csv(error_csv_path, errors_to_update)


# -----------------------
# Runner: N iterations
# -----------------------
if __name__ == "__main__":
    N = 1000

    model_id = "inceptionai/Llama-3.1-Sherkala-8B-Chat"
    temp = 0.2

    tokenizer, model, device = load_model(model_id)

    with open("data/race_genre_language.json", "r", encoding="utf-8") as f:
        lang_data = json.load(f)
    lang = "english"
    n_genres = len(lang_data[lang]["genres"])
    n_ethnicities = len(lang_data[lang]["npc_races"])
    n_npc_genres = len(lang_data[lang]["npc_genres"])
    n_bodies = len(lang_data[lang]["npc_bodies"])
    combinations_per_iter = n_genres * n_ethnicities * n_npc_genres * n_bodies

    total_steps = N * combinations_per_iter

    with tqdm(total=total_steps, unit="npc") as pbar:
        for i in range(N):
            create_npc("npc_system", "npc_user", model_id, temp, i, tokenizer, model, pbar, device)
