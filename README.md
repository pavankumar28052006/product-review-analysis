# Sentiment Analysis System for E-Commerce Reviews

This project implements an **end-to-end sentiment analysis system** for e-commerce product reviews using a **Support Vector Machine (SVM)** model and a **Flask web application**.

## Features

- **Real-time Sentiment Prediction**: Classify reviews as Positive, Negative, or Neutral instantly.
- **User Authentication**: Secure registration and login system.
- **Review History**: Persistent storage of analyzed reviews for each user.
- **Modern UI**: Clean and responsive interface.

## Project Structure

- `app.py`: Flask application core (routes, auth, logic).
- `sentiment_model.py`: ML pipeline (loading, preprocessing, training, saving).
- `database.py`: Database connection and schema management.
- `templates/`: HTML templates for the web interface.
- `static/`: CSS and other static assets.
- `data/`: Directory for the dataset.
- `models/`: Directory for the trained model.

## Setup & Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download NLTK Data**:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

3.  **Prepare Dataset**:
    - Place your dataset in `data/product_reviews.csv`.
    - Ensure it has `review` and `sentiment` columns.

4.  **Train Model**:
    ```bash
    python sentiment_model.py
    ```

5.  **Run Application**:
    ```bash
    # Windows PowerShell
    $env:FLASK_APP="app.py"
    flask run
    ```
    Access the app at `http://127.0.0.1:5000/`.

## Technologies

- **Backend**: Python, Flask, SQLite
- **Machine Learning**: Scikit-learn (SVM, TF-IDF), NLTK, Pandas, NumPy
- **Frontend**: HTML5, CSS3