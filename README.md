# Calorie Prediction Model

A machine learning model that predicts calories per 100g for food items based on their names using natural language processing.

## Project Structure

```
calorie_nlp/
├── data/
│   ├── utils.py         # Text cleaning and feature extraction
│   └── datasets.py      # Dataset loading and preprocessing
├── models/
│   ├── embedder.py      # Sentence transformer model loading
│   ├── mlp.py          # MLP model architecture
│   └── train.py        # Training and evaluation logic
├── experiments/         # Training logs and model checkpoints
├── tests/              # Unit tests
├── config.toml         # Configuration file
└── predict.py          # Command-line interface
```

## Features

- Uses MPNet embeddings for text representation
- Deep MLP architecture with batch normalization and dropout
- Log-transformed calorie predictions
- Huber loss for robust training
- K-fold cross-validation with stratification
- Early stopping to prevent overfitting
- Progress bars for training and validation
- Comprehensive logging of training metrics
- Command-line interface for predictions

## Installation

1. Create a conda environment:
```bash
conda create -n calorie-env python=3.9
conda activate calorie-env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Place your datasets in the `data/` directory:
   - `recipes.csv`: Recipe dataset with columns for name and calories
   - `fruits.csv`: Fruits dataset with columns for name and calories

2. Configure hyperparameters in `config.toml`

3. Run training:
```bash
python -m calorie_nlp.train
```

### Making Predictions

Use the command-line interface to predict calories for food items:

```bash
python predict.py "grilled chicken breast"
```

Optional arguments:
- `--model`: Path to model weights (default: models/best_mlp.pth)
- `--config`: Path to config file (default: config.toml)

## Configuration

The `config.toml` file contains all configurable parameters:

- Model architecture (hidden dimensions, dropout, etc.)
- Training parameters (batch size, learning rate, etc.)
- Data settings (paths, preprocessing options)
- Embedding model configuration

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Features

1. Create a new branch
2. Add tests in `tests/`
3. Implement feature
4. Run tests
5. Submit pull request

## License

MIT License 