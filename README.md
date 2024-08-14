# PDF-RAG-APPLICATION
A simple RAG application to read pdf contains using LLM

### How to run this repo locally
1. Clone this repo

```bash
git clone https://github.com/iamsaptarshi07/PDF-RETRIEVAL-SYSTEM-USING-LLM.git
cd PDF-RETRIEVAL-SYSTEM-USING-LLM
```
2. Create a new virtual environment

```bash
python -m create venv .venv
```

3. Activate the environment

```bash
source .venv/bin/activate
```

4. Install the dependencies

```bash
pip install -r requirements.txt
```

5. Setting up Google API key

- Go to [this](https://aistudio.google.com) website to get Google API key.

- Paste the api key in `.env.example` file and rename it to `.env`.

6. Run the APP

```bash
streamlit run app.py
```
