# File: app.py (Updated for Render NLTK standard path)

import io
import os
import re
import string
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file
import nltk # Make sure nltk is imported
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from werkzeug.utils import secure_filename
import numpy as np
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- REMOVED NLTK Path Manipulation ---
# We now rely on build.sh downloading to a default NLTK search path like /opt/render/nltk_data
logging.info(f"NLTK will search default paths, including: {nltk.data.path}")
# --- End of REMOVED Block ---


# --- Configuration & Constants ---
MODEL_FILENAME = 'hate_speech_model.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'
SENTIMENT_LABELS = {0: 'Not Hate', 1: 'Hate'}
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
ASSUMED_TEXT_COLUMN = 'Content'
script_dir = os.path.dirname(os.path.abspath(__file__)) # Still useful for loading model

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', b'_fallback_secret_')

# --- NLTK Data Check (Relies on Default Paths Now) ---
lemmatizer_instance = None
stop_words_set = set()
try:
    # Test loading stopwords and initialize lemmatizer from default paths
    stop_words_set = set(stopwords.words('english'))
    lemmatizer_instance = WordNetLemmatizer()
    lemmatizer_instance.lemmatize('tests') # Test lemmatization
    logging.info(f"NLTK check successful: Found {len(stop_words_set)} stopwords and lemmatizer initialized from default paths.")
except LookupError as e:
    logging.error("*" * 60)
    logging.error(f"FATAL: Required NLTK data lookup failed: {e}")
    logging.error(f"NLTK searched in these paths: {nltk.data.path}")
    logging.error("Check Render build logs: Did 'python -m nltk.downloader -d /opt/render/nltk_data ...' run successfully?")
    logging.error("Was /opt/render/nltk_data populated correctly during build?")
    logging.error("*" * 60)
    # Consider raising error if NLTK data is critical
    # raise RuntimeError("NLTK data failed to load, cannot continue.")


# --- Load Model and Vectorizer (Path calculation remains the same) ---
model = None
vectorizer = None
model_classes = None
try:
    model_path = os.path.join(script_dir, MODEL_FILENAME)
    vectorizer_path = os.path.join(script_dir, VECTORIZER_FILENAME)
    logging.info(f"Attempting to load model from: {model_path}")
    logging.info(f"Attempting to load vectorizer from: {vectorizer_path}")
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        if hasattr(model, 'classes_'): model_classes = list(model.classes_)
        else: model_classes = [0, 1]; logging.warning("Model has no 'classes_', assuming [0, 1].")
        logging.info(f"Model and vectorizer loaded. Model classes: {model_classes}")
    else:
         logging.error(f"Model ('{model_path}') or Vectorizer ('{vectorizer_path}') file not found.")
         # raise FileNotFoundError("Model or vectorizer file missing.")
except Exception as e:
    logging.exception(f"FATAL: Error loading model/vectorizer: {e}")
    # raise


# --- Text Preprocessing Function ---
def clean_text(text):
    # Use the globally initialized instances
    if lemmatizer_instance is None:
        logging.error("Lemmatizer not initialized, returning original text.")
        return text # Or handle error

    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Use the globally loaded stop_words_set (which might be empty if loading failed)
    cleaned_text = ' '.join([lemmatizer_instance.lemmatize(word) for word in text.split() if word not in stop_words_set])
    return cleaned_text

# --- Helper Function for File Type Check ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes (No changes needed in route logic) ---
@app.route('/', methods=['GET'])
def home():
    logging.info("Serving home page.")
    return render_template('index.html', prediction_text='', probabilities=None, submitted_text='', error_message=None, info_message=None)

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received POST request to /predict.")
    prediction_text = ''
    probabilities_dict = None
    error_message = None
    info_message = None
    submitted_text = request.form.get('text_input', '').strip()
    file = request.files.get('file_upload')

    if model is None or vectorizer is None:
        logging.error("Model or Vectorizer is not loaded. Cannot predict.")
        error_message = "Model or Vectorizer is not available. Cannot perform analysis. Please check server logs."
        return render_template('index.html', error_message=error_message), 503 # Service Unavailable

    # --- BRANCH 1: File Upload Processing ---
    if file and file.filename != '':
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                logging.info(f"Processing uploaded file: {filename}")
                df = None
                if filename.lower().endswith('.csv'):
                     try: df = pd.read_csv(file)
                     except UnicodeDecodeError: file.seek(0); logging.warning("UTF-8 failed for CSV, trying latin-1"); df = pd.read_csv(file, encoding='latin-1')
                elif filename.lower().endswith(('.xlsx', '.xls')): df = pd.read_excel(file, engine='openpyxl')
                else: error_message = "Internal error: Unsupported file format passed filter."; logging.error(f"Unexpected file type allowed: {filename}"); return render_template('index.html', error_message=error_message), 500
                if df is None: error_message = "Failed to read the uploaded file."; logging.error(f"Pandas failed to read file: {filename}"); return render_template('index.html', error_message=error_message), 400
                if ASSUMED_TEXT_COLUMN not in df.columns: error_message = f"Required column '{ASSUMED_TEXT_COLUMN}' not found. Columns: {', '.join(df.columns)}"; logging.warning(f"Missing required column '{ASSUMED_TEXT_COLUMN}' in {filename}."); return render_template('index.html', error_message=error_message), 400 # Bad request

                results = []
                logging.info(f"Found {len(df)} rows to process in {filename}.")
                for index, row in df.iterrows():
                    original_text = row[ASSUMED_TEXT_COLUMN]
                    if pd.isna(original_text) or not isinstance(original_text, str) or original_text.strip() == '': pred_label, prob_not_hate, prob_hate, cleaned_for_output = "Skipped (Invalid/Empty Input)", None, None, ""
                    else:
                        cleaned = clean_text(original_text)
                        vectorized = vectorizer.transform([cleaned])
                        prediction = model.predict(vectorized)[0]
                        proba = model.predict_proba(vectorized)[0]
                        pred_label = SENTIMENT_LABELS.get(prediction, f"Unknown ({prediction})")
                        prob_not_hate, prob_hate = None, None
                        if model_classes and 0 in model_classes: prob_not_hate = proba[model_classes.index(0)]
                        if model_classes and 1 in model_classes: prob_hate = proba[model_classes.index(1)]
                        cleaned_for_output = cleaned
                    results.append({'Original_Text': original_text, 'Cleaned_Text': cleaned_for_output, 'Prediction': pred_label, 'Confidence_Not_Hate': prob_not_hate, 'Confidence_Hate': prob_hate})

                logging.info(f"Finished processing {len(results)} rows from {filename}.")
                results_df = pd.DataFrame(results)
                output_buffer = io.BytesIO(); results_df.to_csv(output_buffer, index=False, encoding='utf-8', float_format='%.4f'); output_buffer.seek(0)
                logging.info(f"Prepared results CSV for download from file {filename}.")
                return send_file(output_buffer, mimetype='text/csv', as_attachment=True, download_name='hate_analysis_results.csv')

            except Exception as e:
                logging.exception(f"Error processing file '{filename}': {e}")
                error_message = f"An error occurred processing the file: {e}"
                return render_template('index.html', error_message=error_message), 500

        else: logging.warning(f"Invalid file type uploaded: {file.filename}"); error_message = "Invalid file type."; return render_template('index.html', error_message=error_message), 400

    # --- BRANCH 2: Single Text Input Processing ---
    elif submitted_text:
        try:
            logging.info(f"Processing single text input (length {len(submitted_text)}).")
            cleaned_input = clean_text(submitted_text)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction_array = model.predict(vectorized_input)
            if prediction_array is not None and len(prediction_array) > 0:
                 prediction = prediction_array[0]; predicted_proba = model.predict_proba(vectorized_input)[0]
                 prediction_text = SENTIMENT_LABELS.get(prediction, f"Unknown ({prediction})")
                 probabilities_dict = {}
                 if model_classes and 0 in model_classes: prob_0 = predicted_proba[model_classes.index(0)]; probabilities_dict['Not Hate'] = f"{prob_0:.1%}"
                 if model_classes and 1 in model_classes: prob_1 = predicted_proba[model_classes.index(1)]; probabilities_dict['Hate'] = f"{prob_1:.1%}"
                 logging.info(f"Single text prediction: {prediction_text}, Probs: {probabilities_dict}")
            else: logging.error("Model prediction array was empty/None."); error_message = "Prediction failed."

        except Exception as e: logging.exception(f"Error during single text prediction: {e}"); error_message = f"An error occurred: {e}"
        status_code = 500 if error_message and not prediction_text else 200
        return render_template('index.html', prediction_text=prediction_text, probabilities=probabilities_dict, submitted_text=submitted_text, error_message=error_message, info_message=None), status_code

    # --- BRANCH 3: No Input Provided ---
    else: logging.warning("Predict called with no input."); error_message = "No input provided."; return render_template('index.html', error_message=error_message), 400

# --- Run the App ---
if __name__ == '__main__':
    if model and vectorizer and lemmatizer_instance:
        port = int(os.environ.get('PORT', 5000))
        logging.info(f"Starting Flask development server on http://0.0.0.0:{port}")
        is_debug = os.environ.get('FLASK_DEBUG', '0') == '1'
        app.run(host='0.0.0.0', port=port, debug=is_debug)
    else:
        logging.error("FATAL: Flask app cannot start - critical components failed to load.")
        import sys; sys.exit(1)