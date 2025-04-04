# File: app.py (Updated for "Hate" / "Not Hate" display)

from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import numpy as np
import pandas as pd
import io
from werkzeug.utils import secure_filename

# --- Configuration & Constants ---
MODEL_FILENAME = 'hate_speech_model.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'
# *** UPDATED LABELS ***
SENTIMENT_LABELS = {0: 'Not Hate', 1: 'Hate'} # Changed labels
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
ASSUMED_TEXT_COLUMN = 'Content'

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# --- NLTK Data Check ---
try:
    stopwords.words('english')
    WordNetLemmatizer().lemmatize('test')
    print("NLTK data found.")
except LookupError:
    print("*" * 60)
    print("ERROR: Required NLTK data not found!")
    print("Please run this command in your terminal:")
    print("python -m nltk.downloader stopwords wordnet omw-1.4")
    print("*" * 60)


# --- Load Model and Vectorizer ---
model = None
vectorizer = None
try:
    if os.path.exists(MODEL_FILENAME) and os.path.exists(VECTORIZER_FILENAME):
        model = joblib.load(MODEL_FILENAME)
        vectorizer = joblib.load(VECTORIZER_FILENAME)
        print("Model and vectorizer loaded successfully.")
    else:
         print(f"ERROR: Model or Vectorizer file not found.")
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")


# --- Text Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
try:
    stop_words_set = set(stopwords.words('english'))
except LookupError:
    print("Stopwords list not found, proceeding without stopword removal.")
    stop_words_set = set()

def clean_text(text):
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words_set])
    return cleaned_text

# --- Helper Function for File Type Check ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction_text='', probabilities=None, submitted_text='', error_message=None, info_message=None)

@app.route('/predict', methods=['POST'])
def predict():
    prediction_text = ''
    probabilities_dict = None
    error_message = None
    info_message = None
    submitted_text = request.form.get('text_input', '').strip()
    file = request.files.get('file_upload')

    if model is None or vectorizer is None:
        error_message = "Model or Vectorizer failed to load. Cannot perform analysis."
        return render_template('index.html', error_message=error_message)

    # --- BRANCH 1: File Upload Processing ---
    if file and file.filename != '':
        if allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                print(f"Processing uploaded file: {filename}")
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file, engine='openpyxl')

                if ASSUMED_TEXT_COLUMN not in df.columns:
                    error_message = f"File processed, but required column '{ASSUMED_TEXT_COLUMN}' not found."
                    return render_template('index.html', error_message=error_message)

                results = []
                print(f"Found {len(df)} rows to process.")
                for index, row in df.iterrows():
                    original_text = row[ASSUMED_TEXT_COLUMN]
                    if pd.isna(original_text) or not isinstance(original_text, str):
                         pred_label = "Skipped (Invalid Input)"
                         prob_not_hate = None
                         prob_hate = None
                    else:
                        cleaned = clean_text(original_text)
                        vectorized = vectorizer.transform([cleaned])
                        prediction = model.predict(vectorized)[0]
                        proba = model.predict_proba(vectorized)[0]
                        pred_label = SENTIMENT_LABELS.get(prediction, "Unknown")

                        # *** UPDATED Probability assignment ***
                        prob_not_hate = proba[list(model.classes_).index(0)] if 0 in model.classes_ else None
                        prob_hate = proba[list(model.classes_).index(1)] if 1 in model.classes_ else None

                    results.append({
                        'Original_Text': original_text,
                        'Prediction': pred_label,
                        # *** UPDATED Probability keys ***
                        'Confidence_Not_Hate': prob_not_hate,
                        'Confidence_Hate': prob_hate
                    })

                print("Finished processing file rows.")
                results_df = pd.DataFrame(results)
                output_buffer = io.BytesIO()
                results_df.to_csv(output_buffer, index=False, encoding='utf-8', float_format='%.4f') # Format probability
                output_buffer.seek(0)

                print("Prepared results CSV for download.")
                return send_file(
                    output_buffer, mimetype='text/csv',
                    as_attachment=True, download_name='hate_analysis_results.csv' # Changed filename
                )

            except Exception as e:
                print(f"Error processing file: {e}")
                error_message = f"An error occurred while processing the file: {e}"
                return render_template('index.html', error_message=error_message)

        else:
            error_message = "Invalid file type. Please upload a CSV, XLSX, or XLS file."
            return render_template('index.html', error_message=error_message)

    # --- BRANCH 2: Single Text Input Processing ---
    elif submitted_text:
        try:
            cleaned_input = clean_text(submitted_text)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction_array = model.predict(vectorized_input)

            if prediction_array is not None and len(prediction_array) > 0:
                 prediction = prediction_array[0]
                 predicted_proba = model.predict_proba(vectorized_input)[0]
                 prediction_text = SENTIMENT_LABELS.get(prediction, "Unknown Classification")

                 # *** UPDATED Probability Dictionary ***
                 probabilities_dict = {}
                 if 0 in model.classes_:
                     probabilities_dict['Not Hate'] = f"{predicted_proba[list(model.classes_).index(0)]:.1%}"
                 if 1 in model.classes_:
                      probabilities_dict['Hate'] = f"{predicted_proba[list(model.classes_).index(1)]:.1%}"

            else:
                error_message = "Prediction failed to return a valid result."

        except Exception as e:
            print(f"Error during single text prediction: {e}")
            error_message = f"An error occurred during analysis. ({e})"

        return render_template('index.html',
                               prediction_text=prediction_text,
                               probabilities=probabilities_dict,
                               submitted_text=submitted_text,
                               error_message=error_message,
                               info_message=None)

    # --- BRANCH 3: No Input ---
    else:
        error_message = "Please enter text in the text area OR upload a file."
        return render_template('index.html', error_message=error_message)


# --- Run the App ---
if __name__ == '__main__':
    if model and vectorizer:
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Flask app not started due to model/vectorizer loading errors.")