# Hate Speech Detection Engine

## Overview

This project is a web application designed to detect potential hate speech in text content. It utilizes a machine learning model trained on labeled data to classify input text as either "Hate" or "Not Hate". The application provides a user-friendly interface for analyzing both single text snippets and text content uploaded via files (CSV/Excel).

## Features

*   **Single Text Analysis:** Paste or type text directly into a text area for immediate analysis.
*   **Batch File Analysis:** Upload CSV, XLSX, or XLS files containing a 'Content' column. The application processes the text in this column and provides predictions. (Note: Current implementation primarily focuses on single text analysis results displayed on the page; file processing might output results differently or require backend adjustment for download).
*   **Classification Output:** Displays the prediction result as either "Hate" or "Not Hate".
*   **Confidence Scores:** Shows the model's confidence probability for both the "Not Hate" and "Hate" classes.
*   **User Feedback:** Provides informative status messages during processing and clear error messages if issues occur (e.g., file format errors, missing data).
*   **Responsive Interface:** The web UI is designed to be usable across different screen sizes.

## How It Works

The project consists of two main parts: the model training pipeline and the Flask web application that serves the model.

### 1. Model Training (`train_model.py`)

This script handles the process of training the hate speech detection model. The key steps involved are:

1.  **Data Loading:**
    *   Loads the training data from `HateSpeechDataset.csv`.
    *   Expects columns named `Content` (for the text) and `Label` (for the classification, where '0' typically represents "Not Hate"/"Neutral" and '1' represents "Hate"/"Offensive").
    *   Includes robust loading with encoding detection and validation checks for columns and missing values.
    *   **Crucially filters the dataset to only include rows where the `Label` is exactly '0' or '1' before converting to integers.**

2.  **Text Preprocessing:**
    *   Applies a series of cleaning steps to the text in the `Content` column to prepare it for the model:
        *   Converts text to lowercase.
        *   Removes punctuation using `string.punctuation`.
        *   Removes numerical digits.
        *   Removes extra whitespace.
        *   Removes common English stopwords (e.g., "the", "a", "is") using `nltk.corpus.stopwords`.
        *   Performs lemmatization (reducing words to their base or dictionary form) using `nltk.stem.WordNetLemmatizer`.
    *   The cleaned text is stored in a new `cleaned_text` column.

3.  **Feature Extraction:**
    *   Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert the preprocessed text into numerical feature vectors.
    *   `TfidfVectorizer` from Scikit-learn is employed, configured with:
        *   `max_features=7000`: Limits the vocabulary size to the top 7000 terms.
        *   `ngram_range=(1, 2)`: Considers both individual words (unigrams) and pairs of adjacent words (bigrams) as features.

4.  **Train-Test Split:**
    *   Splits the TF-IDF features (`X_tfidf`) and the target labels (`y`) into training (80%) and testing (20%) sets.
    *   Uses `stratify=y` to ensure the proportion of "Hate" and "Not Hate" labels is similar in both the training and testing sets, which is important for imbalanced datasets.

5.  **Model Training:**
    *   Trains a **Logistic Regression** classifier (`sklearn.linear_model.LogisticRegression`).
    *   Uses `class_weight='balanced'` to automatically adjust weights inversely proportional to class frequencies, helping the model perform better on imbalanced data.
    *   Uses the `liblinear` solver and increases `max_iter` for convergence.

6.  **Evaluation:**
    *   Evaluates the trained model's performance on the unseen test set using metrics like precision, recall, and F1-score, displayed via a `classification_report`.

7.  **Saving Artifacts:**
    *   Saves the trained Logistic Regression model (`hate_speech_model.pkl`) and the fitted TF-IDF vectorizer (`tfidf_vectorizer.pkl`) using `joblib`. These files are essential for making predictions in the web application.

### 2. Web Application (`app.py`)

This Flask application provides the web interface and uses the trained model to perform predictions.

1.  **Setup:**
    *   Initializes the Flask application.
    *   **Loads the pre-trained model (`hate_speech_model.pkl`) and TF-IDF vectorizer (`tfidf_vectorizer.pkl`)** saved during the training phase. This happens once when the application starts.
    *   **Crucially, it also initializes necessary components for text preprocessing (like the NLTK lemmatizer and stopwords list)**, ensuring consistency with the training script.

2.  **Routing:**
    *   Defines routes for the web application:
        *   `/` (GET): Renders the main HTML page (`templates/index.html`).
        *   `/predict` (POST): Handles the form submission containing either text input or an uploaded file.

3.  **Prediction Logic (within `/predict`):**
    *   Retrieves the text input from the form's text area or reads the 'Content' column from the uploaded file (handling CSV/XLSX/XLS using Pandas).
    *   **Applies the *exact same* Text Preprocessing steps** (lowercasing, punctuation/number removal, stopword removal, lemmatization) to the input text as were used during model training. **This consistency is critical for accurate predictions.**
    *   Transforms the preprocessed input text into a TF-IDF feature vector using the **loaded `vectorizer`**.
    *   Feeds the transformed feature vector into the **loaded `model`**'s `predict_proba` method. This returns the probability for each class (Not Hate, Hate).
    *   Determines the final predicted label ("Not Hate" or "Hate") based on the higher probability.
    *   Formats the prediction label and the probabilities.

4.  **Rendering Results:**
    *   Renders the `index.html` template again, passing the prediction results (label, probabilities), any error/info messages, and the original submitted text back to the user for display.

### 3. Deployment (Render)

*   The application is configured for deployment on platforms like Render.
*   `requirements.txt` lists all necessary Python packages.
*   `gunicorn` is used as the production WSGI server (specified in the Render Start Command: `gunicorn app:app`).
*   A `render-build.sh` script handles build-time tasks, specifically installing dependencies from `requirements.txt` and **downloading the required NLTK data (`stopwords`, `wordnet`, `omw-1.4`)**, ensuring the application has access to these resources when it runs.
*   The saved model (`.pkl`) and vectorizer (`.pkl`) files must be included in the repository to be accessible by the deployed application.

## Running the Application

The application is deployed on Render and can be accessed directly via your browser:

**\[Link to your deployed Render App URL]**

No local setup is required to use the deployed web application.

## File Structure