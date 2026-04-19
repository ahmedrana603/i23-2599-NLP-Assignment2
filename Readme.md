# i23-2599-NLP-Assignment2

**CS-4063: Natural Language Processing — Assignment 2**
**FAST National University of Computer & Emerging Sciences**

A complete Neural NLP Pipeline for Urdu, built entirely from scratch in PyTorch.
This is a continuation of Assignment 1 (BBC Urdu corpus collection and preprocessing).

---

## Repository Structure

```
i23-2599-NLP-Assignment2/
│
├── i23-2599_Assignment2_DS-X.ipynb   ← Main notebook (all cells executed)
├── report.pdf                         ← Assignment report (PDF only)
├── README.md                          ← This file
│
├── embeddings/
│   ├── tfidf_matrix.npy               ← TF-IDF term-document matrix
│   ├── ppmi_matrix.npy                ← PPMI word-word co-occurrence matrix
│   ├── embeddings_w2v.npy             ← Averaged Skip-gram embeddings ½(V+U)
│   └── word2idx.json                  ← Vocabulary word → index mapping
│
├── models/
│   ├── bilstm_pos.pt                  ← Best BiLSTM POS tagger checkpoint
│   ├── bilstm_ner.pt                  ← Best BiLSTM NER model (with CRF)
│   └── transformer_cls.pt             ← Best Transformer topic classifier
│
└── data/
    ├── pos_train.conll                ← POS annotated training data (CoNLL)
    ├── pos_test.conll                 ← POS annotated test data (CoNLL)
    ├── ner_train.conll                ← NER annotated training data (CoNLL)
    └── ner_test.conll                 ← NER annotated test data (CoNLL)
```

> **Note:** `cleaned.txt`, `raw.txt`, and `Metadata.json` from Assignment 1 are
> required to run the notebook but are **not included** in this repository due to
> file size. Place them in the root directory before running.

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Google Colab (recommended) or local GPU environment

Install all dependencies by running the first cell of the notebook, or manually:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy tqdm
```

---

## How to Reproduce

### Option A — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `i23-2599_Assignment2_DS-X.ipynb` via **File → Upload notebook**
3. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU**
4. Run the upload cell and upload your `cleaned.txt`, `raw.txt`, and `Metadata.json`
5. Run all cells top to bottom: **Runtime → Run all**

### Option B — Local Machine

```bash
# 1. Clone the repository
git clone https://github.com/ahmedrana603/i23-2599-NLP-Assignment2.git
cd i23-2599-NLP-Assignment2

# 2. Install dependencies
pip install torch torchvision scikit-learn matplotlib seaborn numpy tqdm

# 3. Place Assignment 1 files in the root directory
#    cleaned.txt, raw.txt, Metadata.json

# 4. Launch Jupyter and run the notebook
jupyter notebook i23-2599_Assignment2_DS-X.ipynb
```

---

## What is Implemented

### Part 1 — Word Embeddings (25 marks)

- **TF-IDF** term-document matrix (vocab capped at 10,000 tokens); top-10 discriminative words per topic reported
- **PPMI** word-word co-occurrence matrix (window k=5); t-SNE visualization of top-200 tokens color-coded by semantic category
- **Skip-gram Word2Vec** trained from scratch with separate center (V) and context (U) matrices, noise distribution f(w)^(3/4), K=10 negative samples, BCE loss, 5 epochs, batch size 512
- **Four-condition comparison** (PPMI baseline, raw corpus, cleaned corpus d=100, cleaned corpus d=200) evaluated with MRR on 20 labeled word pairs
- Analogy tests using vector arithmetic v(b) − v(a) + v(c)

### Part 2 — Sequence Labeling: POS Tagging & NER (25 marks)

- **Rule-based POS tagger** with 200+ hand-crafted lexicon entries and morphological suffix rules; tagset: NOUN VERB ADJ ADV PRON DET CONJ POST NUM PUNC UNK
- **NER gazetteer** covering 50+ Pakistani persons, 50+ locations, 30+ organizations; BIO annotation scheme
- **2-layer Bidirectional LSTM** with dropout=0.5, initialized from Word2Vec embeddings (frozen and fine-tuned modes compared)
- **CRF output layer** with learnable tag-transition matrix; **Viterbi decoding** implemented from scratch (for NER)
- **Ablation study**: A1 unidirectional LSTM, A2 no dropout, A3 random embeddings, A4 softmax vs CRF

### Part 3 — Transformer Encoder for Topic Classification (20 marks)

- **Scaled dot-product attention** with optional padding mask (manual implementation)
- **Multi-head self-attention** (h=4 heads, d_model=128, d_k=d_v=32) with separate projection matrices per head
- **Sinusoidal positional encoding** stored as a fixed non-learned buffer
- **Pre-Layer Normalization** encoder blocks stacked ×4
- **Learned [CLS] token** + MLP classification head (128→64→5)
- **AdamW** optimizer with cosine LR schedule and 50 warmup steps
- Attention weight heatmaps for 3 correctly classified articles (≥2 heads)
- BiLSTM vs Transformer comparative analysis

> ⚠️ **Restrictions respected:** `nn.Transformer`, `nn.MultiheadAttention`, and
> `nn.TransformerEncoder` were NOT used anywhere in this codebase.

---

## Key Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Vocabulary size | — | 10,000 |
| Word2Vec embedding dim (C3) | d | 100 |
| Word2Vec embedding dim (C4) | d | 200 |
| Context window | k | 5 |
| Negative samples | K | 10 |
| Learning rate (Word2Vec) | η | 0.001 (Adam) |
| LSTM hidden dim | H | 128 |
| LSTM layers | — | 2 |
| LSTM dropout | p | 0.5 |
| Transformer d_model | — | 128 |
| Transformer heads | h | 4 |
| Transformer d_ff | — | 512 |
| Transformer layers | — | 4 |
| Max sequence length | — | 256 |
| Learning rate (Transformer) | η | 5×10⁻⁴ (AdamW) |
| Warmup steps | — | 50 |

---

## Output Files Produced by the Notebook

| File | Description |
|------|-------------|
| `embeddings/tfidf_matrix.npy` | TF-IDF matrix (vocab × docs) |
| `embeddings/ppmi_matrix.npy` | PPMI matrix (vocab × vocab) |
| `embeddings/embeddings_w2v.npy` | ½(V+U) averaged embeddings |
| `embeddings/word2idx.json` | Vocabulary mapping |
| `models/bilstm_pos.pt` | POS BiLSTM best checkpoint |
| `models/bilstm_ner.pt` | NER BiLSTM+CRF best checkpoint |
| `models/transformer_cls.pt` | Transformer classifier best checkpoint |
| `data/pos_train.conll` | POS training set (CoNLL format) |
| `data/pos_test.conll` | POS test set (CoNLL format) |
| `data/ner_train.conll` | NER training set (CoNLL format) |
| `data/ner_test.conll` | NER test set (CoNLL format) |

---

## Academic Integrity

All code in this repository is original work. No pretrained models, Gensim, or
HuggingFace libraries were used. Any resemblance to other submissions is coincidental.

---

*CS-4063 Natural Language Processing — FAST NUCES*