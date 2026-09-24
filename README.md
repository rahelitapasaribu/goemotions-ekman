# GoEmotions-Ekman: Multi-label Emotion Classification

## 📋 Project Description
Multi-label Emotion Classification menggunakan dataset GoEmotions-Ekman.
Diberikan satu komentar Reddit berbahasa Inggris, model memprediksi emosi yang berlaku dari 7 label Ekman.

## 🎯 Task
- **Input**: Komentar Reddit (Inggris)
- **Output**: Satu atau lebih label emosi dari: `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, `neutral`
- **Type**: Multi-label Binary Classification
- **Loss**: BCEWithLogitsLoss
- **Model**: bert-base-uncased (fine-tuned)

## 📁 Project Structure
```
goemotions-ekman/
├── data/                    # Data files
├── notebooks/
│   └── eda.ipynb            # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── config.py            # Hyperparameters & paths
│   ├── data_loader.py       # Dataset loading & validation
│   ├── preprocessing.py     # Tokenization & PyTorch Dataset
│   ├── model.py             # Model definition
│   ├── train.py             # Training loop (HuggingFace Trainer)
│   ├── evaluation.py        # Metrics & threshold selection
│   └── inference.py         # Inference pipeline
├── gui/
│   └── app.py               # Gradio GUI for inference
├── main.py                  # Full pipeline entry point
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## 🛠️ Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python main.py
```

### Run GUI
```bash
python gui/app.py
```

### Run EDA
Open `notebooks/eda.ipynb` in Jupyter/VSCode.

## ⚙️ Configuration
All hyperparameters are centralized in `src/config.py`:
- `SEED = 42` (reproducibility)
- `MODEL_NAME = "bert-base-uncased"`
- `MAX_SEQ_LENGTH = 128`
- `LEARNING_RATE = 2e-5`
- `BATCH_SIZE = 16`
- `NUM_EPOCHS = 5`

## 📊 Metrics
- Micro/Macro Precision, Recall, F1
- Per-label F1
- Hamming Loss

## 🔧 Threshold Selection
Global threshold dipilih pada validation set dari kandidat [0.3, 0.35, 0.4, ..., 0.7] berdasarkan Macro-F1.

## 👥 Team
- [Nama anggota kelompok]

## 📝 Academic
- Course: Machine Learning for Text
- Program: Sarjana Informatika, Semester 7
- Year: 2026/2027
