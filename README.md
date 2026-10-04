# StudyBuddy AI

Built solo for **GirlHacks 2026** (NJIT, Enchanted Grove) by Mridhula Polur.

## The problem
Students drowning in lecture PDFs and slide decks default to generic AI chatbots for study help — but those answers aren't grounded in what was actually taught, so you end up fact-checking the AI or studying the wrong thing.

## The idea
Upload your own lecture notes or slides. StudyBuddy AI builds a quiz, grounded entirely in your actual material — every question traceable back to the exact slide or page it came from.

## Why it's different from "just ask ChatGPT"
- **Grounded, not generic** — semantic search (FAISS) over your own uploaded content means every question comes from what your professor actually said, not general web knowledge
- **Visible proof, not just a claim** — every question shows its source location (e.g. "slide 18"), so you can verify it yourself instead of trusting a black box
- **Built on your material, not a pre-made question bank** — works on whatever you uploaded today, not generic flashcard decks

## How it works
1. **Upload** a PDF or PPTX — extracted and split into chunks (one slide = one chunk; PDFs use a sliding window)
2. **Embed** each chunk with Sentence-Transformers (`all-MiniLM-L6-v2`) and index with FAISS for semantic search
3. **Ask** a topic — the top matching chunks are retrieved
4. **Generate** — those chunks are sent to Gemini with an explicit instruction to answer only from the provided material, returning structured, citable quiz questions
5. **Quiz** — answer in the Streamlit UI, get graded instantly with the source cited for each question

## Tech stack
- **Ingestion:** `pdfplumber`, `python-pptx`
- **Retrieval:** Sentence-Transformers embeddings + FAISS vector search
- **Generation:** Gemini API, grounded on retrieved chunks only
- **Frontend:** Streamlit

## Running it locally
```bash
pip install pdfplumber python-pptx sentence-transformers faiss-cpu numpy google-generativeai streamlit
$env:GEMINI_API_KEY = "your-key-here"   # PowerShell
streamlit run app.py
```

## Target categories
GirlHacks 2026: Sprouting Seeds (Beginner) · AI for the Modern Enterprise (ADP)

## Built by
Mridhula Polur · M.S. Computer Science (AI track), Binghamton University
github.com/mpolur-blip · linkedin.com/in/mridhulapolur