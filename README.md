# Calorie NLP Model

A minimal, reproducible machine learning pipeline for predicting calories from food names using NLP techniques and a deep MLP model.

---

## 🚦 Quickstart

Follow these steps to get the project up and running on your machine:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Manas2006/calorie-nlp-model.git
   cd calorie-nlp-model
   ```
2. **(Recommended) Create a virtual environment:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Prepare your data:**
   - Place your `recipes.csv` and/or `fruits.csv` in the `data/` directory (not included in repo).
   - Or use your own dataset with the same format (see `calorie_nlp/data/datasets.py`).
5. **(Optional) Train the model:**
   ```sh
   python3 -m calorie_nlp.train --config config.toml
   ```
   - This will save a model checkpoint to `models/best_mlp.pth`.
6. **Run predictions:**
   - To avoid Hugging Face cache permission issues, use a local cache directory:
   ```sh
   HF_HOME=./model_cache python3 -m calorie_nlp.predict --model-path ./models/best_mlp.pth --input "grilled chicken salad"
   ```
   - Replace the input string and model path as needed.

---

## 🥗 Introduction
This project provides a robust, modular pipeline for predicting the calorie content of foods based on their names. It leverages modern NLP (Sentence-BERT) embeddings and a deep MLP architecture, with a focus on reproducibility, extensibility, and ease of use.

- **No large data or model files are included**—users must provide their own data and/or model checkpoint for training or inference.

---

## 🚀 Features
- Deep MLP model for regression on log-calories
- Sentence-BERT (all-mpnet-base-v2) embeddings for food names
- Structured features (e.g., token count)
- Modular, reproducible codebase
- Sliding-window cross-validation and early stopping
- Configurable via `config.toml`
- CLI for prediction
- Data processing and loading utilities
- Unit tests for core components

---

## 📁 Project Structure
```
calorie_nlp/
├── data/
│   ├── utils.py         # Text cleaning and feature extraction
│   └── datasets.py      # Dataset loading and preprocessing
├── models/
│   ├── embedder.py      # Sentence transformer model loading
│   ├── mlp.py           # MLP model architecture
│   └── train.py         # Training and evaluation logic
├── tests/               # Unit tests
├── predict.py           # CLI for predictions
├── train.py             # Main training script
config.toml              # Configuration file
requirements.txt         # Python dependencies
setup.py                 # Package setup
README.md                # Project documentation
```

---

## 📊 Data Preparation
- Place your `recipes.csv` and/or `fruits.csv` in the `data/` directory (not included in repo).
- Or use your own dataset with the same format (see `calorie_nlp/data/datasets.py` for expected columns and preprocessing).

---

## 🏋️ Training
1. **Configure hyperparameters:**
   - Edit `config.toml` to set model, training, and data parameters.
2. **Train the model:**
   ```sh
   python3 -m calorie_nlp.train --config config.toml
   ```
   - This will train a new model and save the checkpoint to `models/`.

---

## 🔮 Inference / Prediction
Run predictions from the command line:
```sh
HF_HOME=./model_cache python3 -m calorie_nlp.predict --model-path ./models/best_mlp.pth --input "grilled chicken salad"
```
- Replace `models/best_mlp.pth` with your own trained model checkpoint.
- See `calorie_nlp/predict.py` for more CLI options.

---

## 🛠️ Configuration
- All configuration is handled via `config.toml`.
- You can set model architecture, training parameters, data paths, and more.

---

## 🧪 Testing
Run unit tests to verify core functionality:
```sh
pytest calorie_nlp/tests/
```

---

## 🤝 Contributing
Pull requests and issues are welcome! Please open an issue to discuss major changes first.

---

## 📄 License
MIT License. See `LICENSE` file for details. 