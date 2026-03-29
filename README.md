# Geopolitical Event Extraction and Global Impact Analysis

This project extracts geopolitical events and their potential global impact from news articles using NLP and Transformers.

It includes:

- News collection (NewsAPI via `requests`, with a safe sample-data fallback)
- Text preprocessing with spaCy (tokenization, lemmatization, stopword removal)
- Named Entity Recognition for `GPE`, `ORG`, and `PERSON`
- Event type classification using HuggingFace Transformers (with keyword fallback)
- Sentiment polarity scoring (TextBlob)
- A country-focused knowledge graph (NetworkX) with sentiment-weighted edges
- A chronological timeline of extracted events
- Impact scoring for event types
- A Streamlit dashboard to visualize results

## Features

- Clean, modular code (`src/` contains one module per step)
- Beginner-friendly but scalable architecture
- Defensive fallbacks so the pipeline still runs without API keys or model downloads

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Recommended) Install spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## How to Run

### Streamlit dashboard

```bash
cd geopolitical-nlp
streamlit run app.py
```

### CLI run

```bash
cd geopolitical-nlp
python main.py --query "geopolitical conflict sanctions" --limit 10
```

## Configuration

Set your NewsAPI key as an environment variable:

```bash
export NEWSAPI_API_KEY="YOUR_NEWSAPI_KEY"
```

If the API key is missing, the project automatically uses built-in sample articles so the rest of the pipeline can still run.

**Security:** Never commit your real NewsAPI key. Use `NEWSAPI_API_KEY` locally or in CI secrets.

## Git / pushing to GitHub

- Virtual environments (`.venv/`, `.venv311/`, `venv/`) are listed in `.gitignore` and must not be committed.
- After cloning, create a fresh venv and `pip install -r requirements.txt`.

