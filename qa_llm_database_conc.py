import os
import json
import logging
import torch
import sqlite3
import warnings
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# === Config ===
HUGGINGFACE_TOKEN = "your-hf-token"
LARGE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SMALL_MODEL_NAME = "prajjwal1/bert-mini"
DB_FILE_PATH = "qa_database.db"
JSON_FILE_PATH = "questionanswer_data.json"

# === Setup ===
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)

# ✅ Force full core usage
torch.set_num_threads(cpu_count())
torch.set_num_interop_threads(cpu_count())

# === Globals ===
small_tokenizer, small_model = None, None
large_tokenizer, large_model = None, None
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Authenticate Hugging Face ===
def authenticate_huggingface():
    login(token=HUGGINGFACE_TOKEN)
    logging.info("✅ Hugging Face authentication successful.")

# === Create SQLite Table ===
def create_sqlite_table():
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            generated_answer TEXT,
            correct_answer TEXT
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("✅ SQLite table ready.")

# === Save to JSON + SQLite ===
def save_to_file_and_db(question, answer, correct_answer=None):
    qa_data = []
    if os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, "r") as f:
            qa_data = json.load(f)

    qa_data.append({
        "question": question,
        "generated_answer": answer,
        "correct_answer": correct_answer or answer
    })

    with open(JSON_FILE_PATH, "w") as f:
        json.dump(qa_data, f, indent=4)

    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO qa_pairs (question, generated_answer, correct_answer)
        VALUES (?, ?, ?)
    ''', (question, answer, correct_answer or answer))
    conn.commit()
    conn.close()
    logging.info("✅ Saved to DB & JSON.")

# === Load Models in Parallel ===
def load_small_model():
    tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_NAME)
    model = AutoModel.from_pretrained(SMALL_MODEL_NAME)
    logging.info("✅ Small model (BERT Mini) loaded.")
    return tokenizer, model

def load_large_model():
    tokenizer = AutoTokenizer.from_pretrained(LARGE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LARGE_MODEL_NAME,
        device_map={"": "cpu"},
        torch_dtype=torch.float32
    )
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    logging.info("✅ Large model (LLaMA 3 8B) loaded.")
    return tokenizer, model

# === Similarity Check with Full Core Use ===
def check_existing_answer(query, threshold=0.75):
    if not os.path.exists(JSON_FILE_PATH):
        return None, None

    with open(JSON_FILE_PATH, "r") as f:
        qa_data = json.load(f)

    query_emb = embedding_model.encode(query, convert_to_tensor=True)

    def calc_sim(item):
        item_emb = embedding_model.encode(item["question"], convert_to_tensor=True)
        sim = util.pytorch_cos_sim(query_emb, item_emb).item()
        return sim, item

    results = Parallel(n_jobs=cpu_count())(
        delayed(calc_sim)(item) for item in qa_data
    )

    for sim, item in sorted(results, key=lambda x: x[0], reverse=True):
        if sim >= threshold:
            return item["correct_answer"] or item["generated_answer"], item["question"]

    return None, None

# === Generate Answer ===
def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cpu")
    outputs = model.generate(
        **inputs,
        max_length=150,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Main Application ===
def main():
    global small_tokenizer, small_model, large_tokenizer, large_model

    create_sqlite_table()
    authenticate_huggingface()

    # Load models in parallel threads
    with ThreadPoolExecutor() as executor:
        small_future = executor.submit(load_small_model)
        large_future = executor.submit(load_large_model)
        small_tokenizer, small_model = small_future.result()
        large_tokenizer, large_model = large_future.result()

    while True:
        user_prompt = input("\n🧠 Enter your question (or type 'exit' to quit): ").strip()
        if user_prompt.lower() == "exit":
            print("👋 Exiting. Goodbye!")
            break

        # 1. Check similarity using small model (parallelized)
        existing_answer, matched_q = check_existing_answer(user_prompt)
        if existing_answer:
            print(f"\n🟡 Found similar question: '{matched_q}'")
            print(f"💬 Answer: {existing_answer}")
            confirm = input("✅ Is this answer correct? (yes/no): ").strip().lower()
            if confirm == "no":
                correct_answer = input("✏️ Please provide the correct answer: ").strip()
                save_to_file_and_db(user_prompt, existing_answer, correct_answer)
                print("✅ Corrected answer saved.")
            else:
                save_to_file_and_db(user_prompt, existing_answer)
                print("✅ Existing answer confirmed.")
            continue

        # 2. Use large model (LLaMA 3) for new questions
        print("⚙️ Generating answer with LLaMA 3. Please wait...")
        answer = generate_answer(user_prompt, large_tokenizer, large_model)
        print(f"\n🧠 Answer:\n{answer}")
        confirm = input("✅ Is this answer correct? (yes/no): ").strip().lower()
        if confirm == "no":
            correct_answer = input("✏️ Please provide the correct answer: ").strip()
            save_to_file_and_db(user_prompt, answer, correct_answer)
            print("✅ Corrected answer saved.")
        else:
            save_to_file_and_db(user_prompt, answer)
            print("✅ Answer saved.")

if __name__ == "__main__":
    main()
