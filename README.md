🧠 SmartQA – Intelligent Q&A with LLaMA 3 + BERT Memory
SmartQA is a hybrid local question-answering system that combines the speed of a lightweight similarity model with the intelligence of Meta’s LLaMA 3. It checks if a question has already been answered before generating new responses—and learns from user feedback.

Includes both command-line and Streamlit UI interfaces.

🚀 Features
🔍 Checks for similar past questions using Sentence-BERT

🧠 Generates fresh answers using Meta LLaMA-3 8B Instruct

✅ Confirms or corrects answers via user feedback

📦 Stores Q&A pairs in both SQLite and JSON

🌐 Optional web UI using Streamlit

📁 Project Structure

smartqa/
├── smartqa.py           # Main backend logic
├── smartqa_ui.py        # Streamlit-based web interface
├── qa_database.db       # SQLite database (auto-created)
├── questionanswer_data.json  # JSON backup of Q&A (auto-created)
└── README.md
💻 Getting Started
**1. Clone the Repo**

git clone https://github.com/your-username/smartqa.git
cd smartqa

**2. Install Dependencies**
Create a virtual environment (optional but recommended), then:

pip install -r requirements.txt

**3. Set Your Hugging Face Token**
In smartqa.py, replace 'Hugging_face_token' with you own hugging face token
HUGGINGFACE_TOKEN = "your_hf_token_here"
You can get your token from: https://huggingface.co/settings/tokens

🧪 Run the App
▶️ Option 1: Terminal-based Q&A

python smartqa.py
🌐 Option 2: Web UI with Streamlit

streamlit run smartqa_ui.py
