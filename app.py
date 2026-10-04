"""
StudyBuddy AI - Streamlit shell: upload -> retrieve -> generate -> quiz.
Run with: streamlit run app.py
"""

import os
import streamlit as st

from pipeline import extract_pdf, extract_pptx, chunk_document, ChunkIndex
from quiz_generation import generate_quiz

st.set_page_config(page_title="StudyBuddy AI", layout="centered")

# ---------------------------------------------------------------------------
# Session state - keeps the index alive across Streamlit reruns
# ---------------------------------------------------------------------------

if "index" not in st.session_state:
    st.session_state.index = ChunkIndex()
    st.session_state.docs_loaded = []
    st.session_state.quiz = None
    st.session_state.answers = {}

st.title("StudyBuddy AI")
st.caption("Upload your notes. Get quizzes grounded in what you were actually taught.")

if not os.environ.get("GEMINI_API_KEY"):
    st.warning("GEMINI_API_KEY is not set. Set it in your terminal before running this app.")

# ---------------------------------------------------------------------------
# Step 1: Upload -> extract -> chunk -> index
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload lecture notes (PDF) or slides (PPTX)", type=["pdf", "pptx"])

if uploaded_file is not None and uploaded_file.name not in st.session_state.docs_loaded:
    with st.spinner(f"Reading {uploaded_file.name}..."):
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        is_slides = uploaded_file.name.lower().endswith(".pptx")
        pages = extract_pptx(temp_path) if is_slides else extract_pdf(temp_path)
        chunks = chunk_document(pages, source_doc=uploaded_file.name, is_slides=is_slides)

        if st.session_state.index.index is None:
            st.session_state.index.build(chunks)
        else:
            # add_incremental isn't on the lean ChunkIndex - just rebuild with everything
            st.session_state.index.build(st.session_state.index.chunks + chunks)

        st.session_state.docs_loaded.append(uploaded_file.name)

    st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")

if st.session_state.docs_loaded:
    st.write("**Loaded documents:**", ", ".join(st.session_state.docs_loaded))

# ---------------------------------------------------------------------------
# Step 2: Ask a topic -> retrieve grounding chunks -> generate quiz
# ---------------------------------------------------------------------------

st.divider()
topic = st.text_input("What topic do you want to be quizzed on? (separate multiple topics with commas)")

generate_clicked = st.button("Generate quiz", disabled=not st.session_state.docs_loaded)

if generate_clicked:
    if not topic.strip():
        st.error("Enter a topic first.")
    else:
        with st.spinner("Retrieving relevant material and generating quiz..."):
            # Split on commas so each topic gets its own search - otherwise one
            # topic's embedding can dominate and the other gets starved out.
            sub_topics = [t.strip() for t in topic.split(",") if t.strip()]

            seen_ids = set()
            hits = []
            for sub_topic in sub_topics:
                for chunk, score in st.session_state.index.search(sub_topic, top_k=4):
                    if chunk.chunk_id not in seen_ids:
                        seen_ids.add(chunk.chunk_id)
                        hits.append((chunk, score))

            if not hits:
                st.warning("No relevant material found - try a different topic.")
            else:
                # Ask for more questions when covering multiple topics, so each
                # one actually shows up in the quiz instead of getting crowded out.
                num_questions = min(3 * max(1, len(sub_topics)), 6)
                st.session_state.quiz = generate_quiz(topic, hits, num_questions=num_questions)
                st.session_state.answers = {}
# ---------------------------------------------------------------------------
# Step 3: Display quiz, grade on submit
# ---------------------------------------------------------------------------

if st.session_state.quiz:
    st.divider()
    st.subheader("Quiz")

    for i, q in enumerate(st.session_state.quiz):
        st.write(f"**{i + 1}. {q.question}**")
        choice = st.radio(
            label=f"q_{i}",
            options=range(len(q.options)),
            format_func=lambda idx, opts=q.options: opts[idx],
            key=f"radio_{i}",
            label_visibility="collapsed",
        )
        st.session_state.answers[i] = choice

    if st.button("Submit answers"):
        correct_count = 0
        for i, q in enumerate(st.session_state.quiz):
            chosen = st.session_state.answers.get(i)
            was_correct = chosen == q.correct_index
            correct_count += int(was_correct)

            if was_correct:
                st.success(f"Q{i + 1}: correct — {q.explanation} (source: {q.source_location})")
            else:
                st.error(
                    f"Q{i + 1}: incorrect. Correct answer: {q.options[q.correct_index]}. "
                    f"{q.explanation} (source: {q.source_location})"
                )

        st.info(f"Score: {correct_count}/{len(st.session_state.quiz)}")