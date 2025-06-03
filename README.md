# Calorie Prediction Model

A machine learning model that predicts calories in food items using natural language processing. The model uses sentence embeddings and a multi-layer perceptron to predict calories per 100g of food items.

## Features

- Predicts calories per 100g for any food item
- Uses state-of-the-art sentence embeddings (MPNet)
- Handles complex food descriptions
- Provides both CLI and API interfaces
- Supports batch predictions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Manas2006/calorie-nlp-model.git
cd calorie-nlp-model
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Predict calories for a single food item:
```bash
python -m calorie_nlp.predict --input "chocolate milk"
```

Predict calories for multiple food items (default test set):
```bash
python -m calorie_nlp.predict
```

### API Server

Start the FastAPI server:
```bash
python -m calorie_nlp.api
```

The server will start at http://localhost:8000. You can:

1. View API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

2. Make predictions using curl:

   Single food prediction:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"name": "chocolate milk"}'
   ```

   Batch predictions:
   ```bash
   curl -X POST http://localhost:8000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{"items": ["chocolate milk", "apple", "pizza"]}'
   ```

   Health check:
   ```bash
   curl http://localhost:8000/health
   ```

## Model Architecture

The model uses a two-stage approach:
1. Text Embedding: Uses MPNet to generate embeddings for food descriptions
2. Calorie Prediction: A multi-layer perceptron that takes embeddings and additional features to predict calories

## Project Structure

```
calorie_nlp/
├── api.py              # FastAPI server implementation
├── predict.py          # CLI prediction script
├── models/
│   └── mlp.py         # MLP model implementation
├── data/
│   └── utils.py       # Data processing utilities
└── tests/             # Unit tests
```

## Testing

Run the test suite:
```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 