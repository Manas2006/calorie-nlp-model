# Calorie NLP Model

A minimal, reproducible machine learning pipeline for predicting calories from food names using NLP techniques and a deep MLP model.

## Project Overview
- **Purpose:** Predict the calorie content of foods based on their names using modern NLP embeddings and a deep learning model.
- **Tech Stack:** Python, PyTorch, Sentence-BERT, scikit-learn, pandas, numpy.
- **No large data or model files are included**—users must provide their own data and/or model checkpoint for training or inference.

## Features
- Deep MLP model for regression on log-calories
- Sentence-BERT (all-mpnet-base-v2) embeddings for food names
- Structured features (e.g., token count)
- Modular, reproducible codebase
- Sliding-window cross-validation and early stopping
- Configurable via `config.toml`
- CLI for prediction

## Setup
1. **Clone the repository:**
   ```sh
git clone https://github.com/Manas2006/calorie-nlp-model.git
cd calorie-nlp-model
```
2. **Install dependencies:**
   ```sh
pip install -r requirements.txt
```
   Or use your preferred environment manager.

3. **Prepare your data:**
   - Place your `recipes.csv` and/or `fruits.csv` in the `data/` directory (not included in repo).
   - Or use your own dataset with the same format.

4. **(Optional) Train the model:**
   ```sh
python calorie_nlp/train.py --config config.toml
```
   - This will train a new model and save the checkpoint.

5. **Run predictions:**
   ```sh
python calorie_nlp/predict.py --model-path models/best_mlp.pth --input "grilled chicken salad"
```
   - Replace `models/best_mlp.pth` with your own trained model checkpoint.

## Usage
- All configuration is handled via `config.toml`.
- See `calorie_nlp/predict.py` for CLI usage and options.

## Contributing
Pull requests and issues are welcome! Please open an issue to discuss major changes first.

## License
MIT License. See `LICENSE` file for details. 