
import streamlit as st
import cv2
import pytesseract
import numpy as np
import re
import html
from PIL import Image
from pytesseract import Output
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

st.set_page_config(
    page_title="NoteWise AI",
    page_icon="📚",
    layout="wide"
)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0b1020,#17152e);
    color: white;
}
header {visibility:hidden;}
.block-container {
    padding-top:2rem;
    max-width:1200px;
}
.hero {
    background:linear-gradient(120deg,#6338d7,#a14be7);
    padding:30px;
    border-radius:18px;
    margin-bottom:25px;
}
.hero h1 {color:white;font-size:36px;}
.hero p {color:#eee;font-size:16px;}
.card {
    background:#201f38;
    padding:22px;
    border-radius:15px;
    border:1px solid #383453;
    margin-bottom:15px;
}
.metric {
    background:#25223f;
    padding:18px;
    border-radius:12px;
    text-align:center;
}
.metric h2 {color:#bb91ff;}
.stButton>button {
    background:linear-gradient(90deg,#713ce7,#a34ce8);
    color:white;
    border:0;
    border-radius:10px;
    width:100%;
    height:45px;
    font-weight:bold;
}
.stDownloadButton>button {
    background:#713ce7;
    color:white;
    border-radius:10px;
    width:100%;
}
[data-testid="stFileUploader"] {
    background:#201f38;
    border:1px dashed #9873ed;
    padding:15px;
    border-radius:12px;
}
h1,h2,h3 {color:white;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


def process_image(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if gray.shape[1] < 1800:
        scale = 1800 / gray.shape[1]
        gray = cv2.resize(
            gray, None,
            fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    denoised = cv2.bilateralFilter(
        enhanced, 9, 75, 75
    )

    otsu = cv2.threshold(
        denoised, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    adaptive = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )

    candidates = [enhanced, otsu, adaptive]
    best_text = ""
    best_image = enhanced
    best_score = -1

    for candidate in candidates:
        for psm in [6, 11]:

            config = f"--oem 3 --psm {psm}"

            data = pytesseract.image_to_data(
                candidate,
                config=config,
                output_type=Output.DICT
            )

            valid = [
                float(c) for c in data["conf"]
                if str(c) != "-1" and float(c) >= 0
            ]

            confidence = (
                sum(valid) / len(valid) if valid else 0
            )

            text = pytesseract.image_to_string(
                candidate,
                config=config
            ).strip()

            word_count = len(text.split())
            score = confidence + min(word_count, 100) * 0.1

            if text and score > best_score:
                best_score = score
                best_text = text
                best_image = candidate

    return best_image, best_text


def summarize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [
        s.strip() for s in sentences
        if len(s.split()) > 4
    ]

    if not sentences:
        return "Not enough text to summarize."

    if len(sentences) <= 3:
        return " ".join(sentences)

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(sentences)
        scores = np.asarray(matrix.sum(axis=1)).ravel()
        top = np.argsort(scores)[-3:]
        top = sorted(top)

        return " ".join(sentences[i] for i in top)

    except ValueError:
        return " ".join(sentences[:3])


def keywords(text):
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
            ngram_range=(1, 2),
            max_features=100
        )

        matrix = vectorizer.fit_transform([text])
        scores = np.asarray(matrix.sum(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()

        ranked = sorted(
            zip(terms, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:8]

    except ValueError:
        return []


def questions(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    sentences = [
        s.strip() for s in sentences
        if len(s.split()) > 5
    ]

    result = []

    for sentence in sentences[:5]:
        result.append("Explain: " + sentence)

    if not result:
        result = [
            "What is the main topic of these notes?",
            "Explain the important concepts.",
            "Mention the applications of the topic."
        ]

    return result


with st.sidebar:
    st.title("📚 NoteWise AI")
    st.caption("Your Personal Study Assistant")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "About Project"]
    )

    st.markdown("---")
    st.caption("Computer Vision + NLP")
    st.caption("Built with Python")


if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>✨ Study Smarter with AI</h1>
        <p>Scan your notes, extract important information
        and prepare for exams effortlessly.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📤 Upload Your Notes")

    uploaded = st.file_uploader(
        "Choose a notes image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        with st.spinner("Processing your notes..."):
            try:
                processed, extracted_text = process_image(image)

            except Exception as e:
                st.error(f"OCR Error: {e}")
                st.stop()

        left, right = st.columns(2)

        with left:
            st.markdown("#### Original Image")
            st.image(image, use_container_width=True)

        with right:
            st.markdown("#### Processed Image")
            st.image(processed, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📝 Extracted Notes")

        edited_text = st.text_area(
            "Review and edit extracted text:",
            value=extracted_text,
            height=180
        )

        if edited_text.strip():

            summary = summarize(edited_text)
            word_list = keywords(edited_text)
            question_list = questions(edited_text)

            st.markdown("### 📊 Study Insights")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f'<div class="metric"><h2>{len(edited_text.split())}</h2><p>Total Words</p></div>',
                    unsafe_allow_html=True
                )

            with c2:
                st.markdown(
                    f'<div class="metric"><h2>{len(word_list)}</h2><p>Keywords</p></div>',
                    unsafe_allow_html=True
                )

            with c3:
                st.markdown(
                    f'<div class="metric"><h2>{len(question_list)}</h2><p>Questions</p></div>',
                    unsafe_allow_html=True
                )

            st.markdown("### 🧠 Smart Summary")

            st.markdown(
                f'<div class="card">{html.escape(summary)}</div>',
                unsafe_allow_html=True
            )

            st.markdown("### 🔑 Important Keywords")

            if word_list:
                for word, score in word_list:
                    st.write(f"• **{word.capitalize()}**")
            else:
                st.info("No important keywords found.")

            st.markdown("### 🎯 Exam Preparation")

            for i, question in enumerate(question_list, 1):
                st.markdown(f"**Q{i}.** {question}")

            report = f"""
NOTEWISE AI - STUDY REPORT

EXTRACTED NOTES:
{edited_text}

SUMMARY:
{summary}

KEYWORDS:
{word_list}

EXAM QUESTIONS:
{chr(10).join(question_list)}
"""

            st.markdown("---")

            st.download_button(
                "⬇️ Download Study Report",
                report,
                file_name="NoteWise_Study_Report.txt",
                mime="text/plain"
            )

        else:
            st.warning("No readable text found. Try another image.")

    else:
        st.markdown("""
        <div class="card">
            <h3>🚀 How it works</h3>
            <p>1. Upload an image of your notes.</p>
            <p>2. Extract text using OCR.</p>
            <p>3. Generate a smart summary.</p>
            <p>4. Identify important keywords.</p>
            <p>5. Prepare exam questions.</p>
        </div>
        """, unsafe_allow_html=True)


else:

    st.markdown("""
    <div class="hero">
        <h1>About NoteWise AI 📚</h1>
        <p>An intelligent note-scanning and study assistant.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🎯 Project Objective</h3>
        <p>Convert handwritten or printed notes into
        organized digital study material.</p>

        <h3>👁️ Computer Vision</h3>
        <p>OpenCV processes images and Tesseract OCR
        extracts text.</p>

        <h3>🧠 Natural Language Processing</h3>
        <p>TF-IDF performs text summarization and
        keyword extraction.</p>

        <h3>⚙️ Technologies</h3>
        <p>Python | OpenCV | Tesseract | NLP |
        Scikit-learn | Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2026 NoteWise AI | Smart Learning Starts Here")
