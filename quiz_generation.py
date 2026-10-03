"""
StudyBuddy AI - grounded quiz/flashcard generation via the Gemini API.
Takes retrieved chunks from pipeline.ChunkIndex.search() and turns them
into structured, citable quiz questions.
"""

import json
import os
import re
from dataclasses import dataclass

from pipeline import Chunk


# ---------------------------------------------------------------------------
# 1. Output schema
# ---------------------------------------------------------------------------

@dataclass
class QuizQuestion:
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    source_location: str  # e.g. "slide 14" - the grounding citation


# ---------------------------------------------------------------------------
# 2. Prompt construction - the "grounded, not generic" step
# ---------------------------------------------------------------------------

QUIZ_SYSTEM_INSTRUCTIONS = """You are a study assistant. You must generate quiz \
questions ONLY from the provided source material below. Do not use outside \
knowledge, even if you know more about the topic. If the source material is \
too thin to write a good question, write a simpler question rather than \
inventing facts not present in the text.

Return ONLY valid JSON matching this schema, no markdown fences, no preamble:
{
  "questions": [
    {
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 0,
      "explanation": "...",
      "source_location": "..."
    }
  ]
}
"""


def build_grounded_prompt(topic, hits, num_questions=3):
    """hits comes straight from ChunkIndex.search() - (Chunk, score) pairs."""
    context_blocks = [f"[{chunk.location}] {chunk.text}" for chunk, score in hits]
    context = "\n\n".join(context_blocks)

    return f"""{QUIZ_SYSTEM_INSTRUCTIONS}

Topic the student asked about: {topic}

Source material (this is the ONLY thing you may draw questions from):
---
{context}
---

Generate exactly {num_questions} multiple-choice questions. For each question's \
"source_location", copy the bracketed location tag from the block it came from.
"""


# ---------------------------------------------------------------------------
# 3. Gemini call + parsing
# ---------------------------------------------------------------------------

def generate_quiz(topic, hits, num_questions=3):
    """Calls Gemini with the grounded prompt and returns QuizQuestion objects.
    Reads the API key from the GEMINI_API_KEY environment variable."""
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY as an environment variable first.")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-3.8-flash")
    prompt = build_grounded_prompt(topic, hits, num_questions)
    response = model.generate_content(prompt)

    raw_text = response.text.strip()
    raw_text = re.sub(r"^```json|```$", "", raw_text, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}\nRaw response:\n{raw_text}")

    questions = []
    for q in parsed.get("questions", []):
        questions.append(QuizQuestion(
            question=q["question"],
            options=q["options"],
            correct_index=q["correct_index"],
            explanation=q.get("explanation", ""),
            source_location=q.get("source_location", "unknown"),
        ))
    return questions


if __name__ == "__main__":
    # Quick manual test: python quiz_generation.py yourfile.pptx "your topic"
    import sys
    from pipeline import extract_pdf, extract_pptx, chunk_document, ChunkIndex

    if len(sys.argv) > 2:
        path, topic = sys.argv[1], sys.argv[2]
        is_slides = path.lower().endswith(".pptx")
        pages = extract_pptx(path) if is_slides else extract_pdf(path)
        chunks = chunk_document(pages, source_doc=path, is_slides=is_slides)

        index = ChunkIndex()
        index.build(chunks)
        hits = index.search(topic, top_k=4)

        print(f"Top matches for '{topic}':")
        for chunk, score in hits:
            print(f"  [{chunk.location}] score={score:.2f}")

        print("\nGenerating quiz...")
        quiz = generate_quiz(topic, hits, num_questions=3)
        for i, q in enumerate(quiz, 1):
            print(f"\n{i}. {q.question}")
            for j, opt in enumerate(q.options):
                marker = "✓" if j == q.correct_index else " "
                print(f"   [{marker}] {opt}")
            print(f"   Source: {q.source_location}")