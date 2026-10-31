# 📚 NoteWise AI – Smart Notes Scanner & AI Study Assistant

NoteWise AI is a Python-based application that helps students convert their study notes into digital text and useful learning materials using Computer Vision and Natural Language Processing (NLP).

It extracts text from uploaded notes, generates summaries, identifies important keywords and creates practice questions to support effective learning and exam preparation.

## ✨ Features

- **Smart Notes Scanner:** Upload printed or handwritten notes.
- **Image Processing:** Enhance images using OpenCV.
- **Text Extraction:** Extract text using Tesseract OCR.
- **Automatic Summarization:** Generate concise summaries using TF-IDF.
- **Keyword Extraction:** Identify important terms from notes.
- **Practice Questions:** Generate questions from extracted content.
- **MCQ Quiz:** Practice with multiple-choice questions.
- **Revision Flashcards:** Revise important concepts.
- **Download Report:** Download extracted study content.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| OpenCV | Image preprocessing |
| Tesseract OCR | Text recognition |
| NumPy | Image processing operations |
| Pillow | Image handling |
| Scikit-learn | TF-IDF and text analysis |

## 📂 Project Structure

```text
Smart_Notes_AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── screenshots/
└── uploads/
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/SOWJANYA-code53/NoteWise-AI.git
```

### 2. Open the project folder

```bash
cd NoteWise-AI
```

### 3. Install the required libraries

```bash
python -m pip install -r requirements.txt
```

### 4. Install Tesseract OCR

Install Tesseract OCR and configure its executable path in `app.py`.

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🔍 How It Works

1. Upload an image of your study notes.
2. OpenCV preprocesses the image.
3. Tesseract OCR extracts the text.
4. NLP techniques summarize the extracted content.
5. Important keywords and practice questions are generated.
6. Use flashcards and quizzes for revision.
7. Download your study report.

## ⚠️ Limitations

- OCR accuracy depends on image quality and handwriting clarity.
- Tesseract generally performs better with printed text.
- Summarization is extractive and uses TF-IDF.
- Practice questions are generated using rule-based techniques.

## 🎯 Objective

To make studying easier by converting traditional notes into digital, organized and interactive learning materials.

## 👩‍💻 Developer

**Sowjanya Kongani**

GitHub: [SOWJANYA-code53](https://github.com/SOWJANYA-code53)

---

⭐ If you find this project interesting, consider giving the repository a star!