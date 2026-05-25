# Cranfield Search Engine: Vector Space Model & RAG

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Railway-blueviolet?style=for-the-badge)](https://cranfield-search.up.railway.app/)

A modern, high-performance Information Retrieval (IR) System built on the classic **Cranfield Collection** dataset. The system integrates a custom-built **Vector Space Model (VSM)** employing the **`lnc.ltc`** weighting scheme for precise document ranking, combined with a **Retrieval-Augmented Generation (RAG)** pipeline powered by Google's **Gemini 3 Flash** to synthesize direct, concise answers from the top retrieved documents.

The project features both a Command Line Interface (CLI) for batch evaluation and a premium, responsive Web Interface with glassmorphism design aesthetics.

👉 **Try the Live App here:** [https://cranfield-search.up.railway.app/](https://cranfield-search.up.railway.app/)


---

## 🚀 Key Features

*   **Custom Vector Space Model (VSM)**:
    *   **Document weighting (`lnc`)**: Logarithmic term frequency ($l$), no collection frequency weighting ($n$), and cosine normalization ($c$).
    *   **Query weighting (`ltc`)**: Logarithmic term frequency ($l$), inverse document frequency ($t$), and cosine normalization ($c$).
*   **Natural Language Processing**: Preprocessing pipeline utilizing `NLTK` for lowercasing, punctuation removal, word tokenization, stopword filtering, and Porter Stemming.
*   **Automated Evaluation Suite**: Implements Mean Average Precision (**MAP**) and Normalized Discounted Cumulative Gain (**NDCG@10**) using the official Cranfield relevance judgments (`cranqrel`).
*   **Retrieval-Augmented Generation (RAG)**: Connects to the new `google-genai` SDK using `gemini-3-flash-preview` to formulate a 2-3 sentence summary/answer based on the top-ranking documents retrieved.
*   **Premium Web UI**:
    *   Modern dark mode interface with glassmorphism, radial gradient glows, and fluid micro-animations.
    *   Interactive dropdown containing 30 sample queries with their official ground truth relevance judgments mapped in real-time.
    *   Displays tokenized search queries dynamically.
    *   Non-blocking asynchronous AI answer generation powered by Python background threads.

---

## 📁 Project Structure

```text
├── dataset/
│   ├── cran.all.1400      # 1400 Cranfield documents (titles + abstracts)
│   ├── cran.qry           # 225 Cranfield queries
│   └── cranqrel           # Query-document relevance judgments
├── src/
│   ├── parser.py          # Parsers for docs, queries, and qrels (SMART format)
│   ├── preprocess.py      # Tokenization, stopword removal, and Porter Stemming
│   ├── vsm.py             # lnc.ltc Vector Space Model indexing and scoring
│   ├── evaluate.py        # Evaluation metrics calculations (MAP & NDCG)
│   └── rag.py             # Retrieval-augmented generation using Gemini
├── web/
│   └── index.html         # Rich, single-page application frontend
├── app.py                 # Flask server backend serving APIs and static Web UI
├── main.py                # Command line evaluation utility
├── requirements.txt       # Python dependencies list
└── .env                   # Environment variables (API Keys) - Create manually
```

---

## 🛠️ Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10** or higher installed on your system.

### 2. Setup & Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/bequdah/cranfield-search-engine.git
   cd cranfield-search-engine
   ```

2. **Create a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Gemini API Key**:
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   FLASK_ENV=development
   ```

---

## ⚙️ How to Run

### Command Line Evaluation (CLI)
To build the index, run all Cranfield queries, compute overall system performance (MAP & NDCG@10), and view test outputs for Query ID 7:
```bash
python main.py
```

### Web Application
To launch the interactive search engine client locally:
```bash
python app.py
```
After running, open your browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

Or access the globally deployed version directly at:
👉 **[https://cranfield-search.up.railway.app/](https://cranfield-search.up.railway.app/)**


---

## 📊 Evaluation Metrics Details

*   **Mean Average Precision (MAP)**: Measures the quality of binary relevance (relevance levels 1, 2, 3, 4 are considered relevant; missing/negative are non-relevant). Evaluated strictly over queries containing relevant documents.
*   **NDCG@10**: Evaluates graded relevance ranking. Cranfield relevance grades ($1, 2, 3, 4$) are converted to gains ($4, 3, 2, 1$ respectively, $0$ for others) to penalize placing less relevant documents at the top of the results page.

---

## 🏆 System Performance & Results

Below are the actual performance metrics of the VSM model (`lnc.ltc`) evaluated against the entire Cranfield collection (1,400 documents, 225 queries, and 1,837 relevance judgments):

| Metric | Score | Details |
| :--- | :--- | :--- |
| **MAP** | **0.3128** | Mean Average Precision across all queries |
| **NDCG@10** | **0.3751** | Normalized Discounted Cumulative Gain at rank 10 |

### 🔍 Sample Query Test (Query ID 7)
*   **Query**: *"is it possible to relate the available pressure distributions for an ogive forebody at zero angle of attack to the lower surface pressures of an equivalent ogive forebody at angle of attack ."*
*   **Top-10 Precision**: **0.20** (Matches in Top-10: `56`, `57` out of the 5 ground truth relevant documents).


