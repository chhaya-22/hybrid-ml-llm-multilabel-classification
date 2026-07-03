# Multi-Label Text Classification: Classical ML vs Hybrid ML + LLM vs Pure LLM

A comparative study of three approaches to **multi-label text classification** on a product-attribute dataset, showing how a **hybrid Machine Learning + LLM** pipeline handles class imbalance and rare labels better than either approach alone.

---

## 📌 Problem

Each product item (e.g. a body-wash product/review) must be tagged with **one or more** "Level 1 Factor" labels. This is a **multi-label** problem with a key challenge: the label distribution is **highly imbalanced**, so several labels appear in fewer than 5% of samples and are easily missed by a classical model.

---

## 🧪 Approaches compared

### 1. Classical Machine Learning
- **Features:** TF-IDF (`max_features=5000`)
- **Models (4 variants):** Logistic Regression and Linear SVC, each with **OneVsRest** and **ClassifierChain**
- **Evaluation:** F1-micro, Hamming Loss, Jaccard score (on a held-out validation split)
- **Best model:** Logistic Regression + ClassifierChain
- **Limitation found:** 18 of 127 test rows received **blank predictions** because their true labels were rare — the classical model couldn't capture them.

### 2. Hybrid ML + LLM  ⭐ (main contribution)
- Use the **best classical model** for confident predictions.
- Route the **blank / rare-label rows to an LLM** — `llama-3.3-70b-versatile` via the **Groq API**, using **few-shot examples** and `temperature=0` for deterministic output.
- **Result:** no blank predictions, better coverage of rare labels, and improved F1-micro over classical-only.

### 3. Pure LLM
- Few-shot LLaMA-3.3-70B for **all** predictions, as a comparison baseline.

---

## 📊 Key takeaway
The **hybrid pipeline** combines the precision and speed of classical ML on common labels with the LLM's ability to reason about **rare / unseen labels from just a few examples** — giving the most balanced overall performance.

---

## 🛠️ Tech stack
`Python` · `scikit-learn` (TF-IDF, OneVsRest, ClassifierChain) · `Groq API` (LLaMA-3.3-70B) · `pandas` · `matplotlib`

---

## ▶️ How to run

```bash
pip install -r requirements.txt      # pandas, scikit-learn, groq, tqdm, matplotlib, openpyxl

# set your Groq API key as an environment variable (do NOT hardcode it)
export GROQ_API_KEY="your_key_here"    # Windows: setx GROQ_API_KEY "your_key_here"

jupyter notebook NLP_Multilabel_Classification.ipynb
```

> **Note:** the notebook reads the key from `os.environ["GROQ_API_KEY"]`. Never commit an API key to the repository.

---

## 📁 Files
- `NLP_Multilabel_Classification.ipynb` — full pipeline: preprocessing → classical ML → hybrid ML+LLM → pure LLM
- `bodywash-train.xlsx` / `bodywash-test.xlsx` — dataset
- output CSVs — predictions from each approach

---

## ✍️ Author
**Chhaya Gupta** — M.Tech, Data Analytics, IIT (ISM) Dhanbad
