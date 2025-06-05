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
**2. Install Dependencies **
pip install -r requirements.txt

**3. Set Your Hugging Face Token**
In the script file (qa_llm_dtaabase_and_conc.py), replace the default token with your own from Hugging Face:

HUGGINGFACE_TOKEN = "your_hf_token_here"
Generate one here: https://huggingface.co/settings/tokens

▶️ Running the Script
Launch the command-line interface:

python qa_llm_dtaabase_and_conc.py
You’ll be prompted to enter questions and confirm or correct answers, which are saved locally for future reuse.
