🧠 SmartQA – Intelligent Q&A with LLaMA 3 + BERT Memory
SmartQA is a local question-answering system that intelligently combines a fast similarity check using BERT with answer generation via Meta’s LLaMA 3 model. It stores and reuses previous Q&A pairs and allows users to validate or correct answers, continuously improving its usefulness.

🚀 Features
🔍 Similarity Matching: Checks if a similar question has been answered before using Sentence-BERT.

🧠 Answer Generation: Uses Meta LLaMA-3 8B Instruct model for generating new responses.

✅ Learning via Feedback: Confirms or corrects answers with user input.

🗂️ Persistent Storage: Saves all Q&A pairs in both SQLite (qa_database.db) and JSON (questionanswer_data.json) formats.

⚙️ Full CPU Utilization: Uses all available cores for fast embedding similarity comparisons and parallel model loading.

📁 Project Structure
smartqa/

├── qa_llm_dtaabase_and_conc.py    # Core backend logic (this file)

├── qa_database.db                 # SQLite database (auto-created)

├── questionanswer_data.json       # JSON backup of Q&A (auto-created)

└── README.md
💻 Getting Started

**1. Clone the Repo**
git clone https://github.com/your-username/smartqa.git
cd smartqa

**2.Install Dependencies**
pip install -r requirements.txt

**3. Set Your Hugging Face Token**
In the script file (qa_llm_dtaabase_and_conc.py), replace the default token with your own from Hugging Face:

HUGGINGFACE_TOKEN = "your_hf_token_here"
Generate one here: https://huggingface.co/settings/tokens

▶️ Running the Script
Launch the command-line interface:

python qa_llm_dtaabase_and_conc.py
You’ll be prompted to enter questions and confirm or correct answers, which are saved locally for future reuse.

Case 1: If User asks already existing Question
![image](https://github.com/user-attachments/assets/95e2c325-91f7-47e7-87ea-a846b820b4af)

Case 2: If User asks a new Question and AI answers it with wrong context
![image](https://github.com/user-attachments/assets/235a899f-7a18-463c-845f-6159f380808c)

Case 3: If User asks a new Question that is not in DB and AI answers it correctly

![image](https://github.com/user-attachments/assets/26560d93-a085-45ca-b179-5c20442a949a)


**Author**
Created by Supriya Rao

