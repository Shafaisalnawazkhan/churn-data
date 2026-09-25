# Customer Churn Prediction

A Flask web application that predicts whether a customer is likely to churn, backed by a scikit-learn Random Forest model.

## Project files

- `app.py` - Flask application and prediction endpoint.
- `train_model.py` - training script for the churn model.
- `customer_churn_dataset.csv` - training dataset.
- `churn_model.pkl` - exported, trained model pipeline.
- `customer_churn.ipynb` - exploratory notebook.
- `templates/index.html` - web interface.

## Requirements

Use Python 3. Install the dependencies:

```bash
pip install flask pandas scikit-learn joblib
```

## Run the app

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser, enter the customer information, and submit the form to receive a churn prediction.

## Train the model again

The included `churn_model.pkl` is ready for the web app. To recreate it from the CSV dataset:

```bash
python train_model.py
```

The script cleans the data, one-hot encodes categorical columns, trains a Random Forest classifier, displays its test accuracy, and saves the updated model to `churn_model.pkl`.
