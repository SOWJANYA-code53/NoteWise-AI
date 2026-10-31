
import streamlit as st
import cv2
import pytesseract
import numpy as np
import re
import random
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from collections import Counter

st.set_page_config(page_title="NoteWise AI", page_icon="📚", layout="wide")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#101426,#202044,#17152f);
    color: white;
}
[data-testid="stSidebar"] {
    background: #17172d;
}
.hero {
    background: linear-gradient(110deg,#6546e8,#b13fc5);
    padding: 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.panel {
    background: #252641;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #48466e;
    margin-bottom: 15px;
}
.metric {
    background: #29294b;
    padding: 18px;
    border-radius: 13px;
    text-align: center;
}
h1,h2,h3 {
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg,#7655ed,#b54acb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

def preprocess_image(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if gray.shape[1] < 1800:
        scale = 1800 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )

    return [denoised, otsu, adaptive]

def extract_text(image):
    candidates = preprocess_image(image)
    best_text = ""
    best_image = candidates[0]
    best_score = -1
    best_confidence = 0

    for candidate in candidates:
        for psm in [6, 11, 3]:
            config = f"--oem 3 --psm {psm}"

            data = pytesseract.image_to_data(
                candidate,
                config=config,
                output_type=pytesseract.Output.DICT
            )

            words = []
            confidences = []

            for word, confidence in zip(data["text"], data["conf"]):
                word = word.strip()
                try:
                    confidence = float(confidence)
                except:
                    confidence = -1

                if word and confidence >= 0:
                    words.append(word)
                    confidences.append(confidence)

            text = pytesseract.image_to_string(candidate, config=config).strip()

            if not text:
                continue

            average_confidence = (
                sum(confidences) / len(confidences)
                if confidences else 0
            )

            word_count = len(words)
            score = average_confidence + min(word_count, 150) * 0.08

            if score > best_score:
                best_score = score
                best_text = text
                best_image = candidate
                best_confidence = average_confidence

    return best_text, best_image, round(best_confidence, 1)

def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def get_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if len(sentence.split()) >= 5
    ]

def summarize(text, level):
    sentences = get_sentences(text)

    if not sentences:
        return text[:500]

    if len(sentences) <= 2:
        return " ".join(sentences)

    count = {"Short": 2, "Medium": 4, "Detailed": 6}[level]
    count = min(count, len(sentences))

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
        matrix = vectorizer.fit_transform(sentences)
        scores = np.asarray(matrix.sum(axis=1)).flatten()
        selected = sorted(
            np.argsort(scores)[-count:].tolist()
        )
        return " ".join(sentences[i] for i in selected)
    except:
        return " ".join(sentences[:count])

def get_keywords(text, count=10):
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
            ngram_range=(1, 2),
            max_features=150
        )
        matrix = vectorizer.fit_transform([text])
        scores = np.asarray(matrix.sum(axis=0)).flatten()
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(
            zip(terms, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [word for word, score in ranked[:count]]
    except:
        return []

def generate_questions(text):
    sentences = get_sentences(text)
    questions = []

    for sentence in sentences[:10]:
        questions.append("Explain: " + sentence)

    return questions[:8]

def generate_mcqs(text):
    sentences = get_sentences(text)
    words = get_keywords(text, 20)
    mcqs = []

    for sentence in sentences:
        matching = [
            word for word in words
            if re.search(r"\b" + re.escape(word) + r"\b", sentence, re.I)
        ]

        if not matching:
            continue

        answer = max(matching, key=len)
        question = re.sub(
            r"\b" + re.escape(answer) + r"\b",
            "________",
            sentence,
            count=1,
            flags=re.I
        )

        distractors = [
            word for word in words
            if word.lower() != answer.lower()
        ]

        if len(distractors) < 3:
            continue

        options = random.sample(distractors, 3) + [answer]
        random.shuffle(options)

        mcqs.append({
            "question": question,
            "options": options,
            "answer": answer
        })

        if len(mcqs) == 5:
            break

    return mcqs

def generate_flashcards(text):
    words = get_keywords(text, 12)
    sentences = get_sentences(text)
    cards = []

    for word in words:
        matching = [
            sentence for sentence in sentences
            if re.search(r"\b" + re.escape(word) + r"\b", sentence, re.I)
        ]

        if matching:
            cards.append({
                "front": word.title(),
                "back": matching[0]
            })

    return cards

def create_report(text, summary, keywords, questions, mcqs, cards):
    report = "NOTEWISE AI - STUDY REPORT\n\n"
    report += "EXTRACTED NOTES\n" + text + "\n\n"
    report += "SUMMARY\n" + summary + "\n\n"
    report += "IMPORTANT KEYWORDS\n" + ", ".join(keywords) + "\n\n"
    report += "PRACTICE QUESTIONS\n"
    report += "\n".join(questions) + "\n\n"
    report += "MCQ QUESTIONS\n"

    for i, mcq in enumerate(mcqs, 1):
        report += f"\n{i}. {mcq['question']}\n"
        for option in mcq["options"]:
            report += f"- {option}\n"
        report += f"Answer: {mcq['answer']}\n"

    report += "\nFLASHCARDS\n"
    for card in cards:
        report += f"\n{card['front']}: {card['back']}\n"

    return report

if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""

if "processed_image" not in st.session_state:
    st.session_state.processed_image = None

if "ocr_confidence" not in st.session_state:
    st.session_state.ocr_confidence = 0

if "flashcard_index" not in st.session_state:
    st.session_state.flashcard_index = 0

st.sidebar.title("📚 NoteWise AI")
st.sidebar.caption("Your Personal Study Assistant")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "Study Assistant", "About Project"]
)

st.sidebar.divider()
st.sidebar.caption("Computer Vision + NLP")
st.sidebar.caption("Built with Python")

if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>✨ NoteWise AI</h1>
        <h3>Study Smarter with AI</h3>
        <p>Convert your notes into summaries, keywords, questions and revision material.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📤 Upload Your Notes")

    uploaded_file = st.file_uploader(
        "Upload a printed or handwritten notes image",
        type=["png", "jpg", "jpeg", "bmp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        if st.button("🚀 Scan and Analyze Notes"):
            with st.spinner("Enhancing image and extracting text..."):
                try:
                    text, processed, confidence = extract_text(image)
                    st.session_state.notes_text = text
                    st.session_state.processed_image = processed
                    st.session_state.ocr_confidence = confidence
                    st.session_state.flashcard_index = 0
                    st.success("Image processing completed!")
                except Exception as error:
                    st.error(f"Processing failed: {error}")

        if st.session_state.processed_image is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Original Notes")
                st.image(image, use_container_width=True)

            with col2:
                st.markdown("### Processed Notes")
                st.image(
                    st.session_state.processed_image,
                    clamp=True,
                    use_container_width=True
                )

    if st.session_state.notes_text:
        st.divider()
        st.subheader("📊 Study Overview")

        words = st.session_state.notes_text.split()
        keywords = get_keywords(st.session_state.notes_text)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f'<div class="metric"><h2>{len(words)}</h2>Words Extracted</div>',
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f'<div class="metric"><h2>{len(keywords)}</h2>Keywords Found</div>',
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f'<div class="metric"><h2>{st.session_state.ocr_confidence}%</h2>OCR Confidence</div>',
                unsafe_allow_html=True
            )

        st.caption("OCR confidence is an estimate from Tesseract, not a guarantee of text accuracy.")

        st.subheader("📝 Extracted Text")
        st.session_state.notes_text = st.text_area(
            "Edit the extracted notes if needed",
            value=st.session_state.notes_text,
            height=250
        )

        st.info("You can correct OCR mistakes here before generating your study material.")

        if st.button("Generate Study Material"):
            st.session_state.generate = True

        if st.session_state.get("generate", False):
            st.success("Study material is ready. Open Study Assistant from the sidebar.")

elif page == "Study Assistant":
    st.markdown("""
    <div class="hero">
        <h1>🧠 AI Study Assistant</h1>
        <p>Learn, revise and test your knowledge from your notes.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.notes_text:
        st.warning("Please upload and scan your notes from the Dashboard first.")
    else:
        text = st.session_state.notes_text
        summary_tab, keyword_tab, question_tab, mcq_tab, flashcard_tab, export_tab = st.tabs([
            "Summary",
            "Keywords",
            "Questions",
            "MCQ Quiz",
            "Flashcards",
            "Download"
        ])

        with summary_tab:
            st.subheader("📖 Smart Summary")
            level = st.selectbox(
                "Choose summary length",
                ["Short", "Medium", "Detailed"]
            )
            st.markdown(
                '<div class="panel">', unsafe_allow_html=True
            )
            st.write(summarize(text, level))
            st.markdown("</div>", unsafe_allow_html=True)

        with keyword_tab:
            st.subheader("🔑 Important Keywords")
            keywords = get_keywords(text, 15)
            if keywords:
                st.write(" ".join([f"`{word}`" for word in keywords]))
            else:
                st.info("Not enough text to identify keywords.")

        with question_tab:
            st.subheader("✍️ Practice Questions")
            questions = generate_questions(text)

            if questions:
                for i, question in enumerate(questions, 1):
                    st.write(f"**{i}. {question}**")
            else:
                st.info("Not enough complete sentences to generate questions.")

        with mcq_tab:
            st.subheader("🎯 Multiple Choice Quiz")
            mcqs = generate_mcqs(text)

            if mcqs:
                answers = {}

                for i, mcq in enumerate(mcqs):
                    st.write(f"**Q{i+1}. {mcq['question']}**")
                    answers[i] = st.radio(
                        "Choose your answer",
                        mcq["options"],
                        key=f"mcq_{i}",
                        index=None
                    )

                if st.button("Check My Answers"):
                    score = 0

                    for i, mcq in enumerate(mcqs):
                        if answers[i] == mcq["answer"]:
                            score += 1
                            st.success(f"Question {i+1}: Correct!")
                        else:
                            st.error(
                                f"Question {i+1}: Correct answer is {mcq['answer']}"
                            )

                    st.metric("Your Score", f"{score}/{len(mcqs)}")
            else:
                st.info("Add more meaningful notes to generate MCQs.")

        with flashcard_tab:
            st.subheader("🗂️ Revision Flashcards")
            cards = generate_flashcards(text)

            if cards:
                index = st.session_state.flashcard_index % len(cards)
                card = cards[index]

                st.markdown(
                    f'<div class="hero"><h2>{card["front"]}</h2><p>Think about the meaning of this concept.</p></div>',
                    unsafe_allow_html=True
                )

                if st.button("Show Answer"):
                    st.info(card["back"])

                previous, next_card = st.columns(2)

                with previous:
                    if st.button("⬅ Previous"):
                        st.session_state.flashcard_index = (
                            st.session_state.flashcard_index - 1
                        ) % len(cards)
                        st.rerun()

                with next_card:
                    if st.button("Next ➡"):
                        st.session_state.flashcard_index = (
                            st.session_state.flashcard_index + 1
                        ) % len(cards)
                        st.rerun()

                st.caption(f"Card {index+1} of {len(cards)}")
            else:
                st.info("Not enough text to create flashcards.")

        with export_tab:
            st.subheader("📥 Download Study Report")
            summary = summarize(text, "Detailed")
            keywords = get_keywords(text, 15)
            questions = generate_questions(text)
            mcqs = generate_mcqs(text)
            cards = generate_flashcards(text)

            report = create_report(
                text, summary, keywords, questions, mcqs, cards
            )

            st.download_button(
                "Download Complete Study Report",
                data=report,
                file_name="NoteWise_AI_Study_Report.txt",
                mime="text/plain"
            )

elif page == "About Project":
    st.markdown("""
    <div class="hero">
        <h1>About NoteWise AI</h1>
        <p>A Smart Notes Scanner and AI Study Assistant.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🎯 Project Objective")
    st.write(
        "To convert printed or handwritten study notes into digital text "
        "and generate useful learning materials."
    )

    st.subheader("⚙️ Technologies Used")

    st.markdown("""
    - **Python:** Main programming language
    - **OpenCV:** Image preprocessing and enhancement
    - **Tesseract OCR:** Text extraction
    - **NumPy:** Image processing operations
    - **Scikit-learn:** TF-IDF-based keyword extraction and summarization
    - **Streamlit:** Interactive web application
    """)

    st.subheader("🚀 Main Features")

    st.markdown("""
    1. Notes image scanning
    2. Image enhancement
    3. Text extraction
    4. Automatic summaries
    5. Important keyword extraction
    6. Practice question generation
    7. MCQ quiz
    8. Revision flashcards
    9. Study report download
    """)

st.divider()
st.caption("© 2026 NoteWise AI | Built with Python and AI")
