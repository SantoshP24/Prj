# File: app.py (Updated for Render NLTK path and version alignment)

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
import numpy as np # Import numpy for checking model classes

# --- Tell NLTK where to find the data downloaded by build.sh --- START ---
# Construct the absolute path to the nltk_data directory relative to app.py
# __file__ gives the path of the current script (app.py)
# os.path.dirname gets the directory containing the script
# os.path.join combines the directory path and 'nltk_data' correctly
# Use '.' as fallback dirname if __file__ is not available (e.g., interactive)
script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
nltk_data_dir = os.path.join(script_dir, 'nltk_data')

# Check if the directory exists (it should if build.sh ran correctly)
if os.path.exists(nltk_data_dir) and os.path.isdir(nltk_data_dir):
    # Add the directory to NLTK's data path list (if not already present)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
        print(f"Added '{nltk_data_dir}' to NLTK data path.")
    else:
        print(f"NLTK data path '{nltk_data_dir}' already present.")
    # Verify immediately if the core data can be accessed
    try:
        nltk.corpus.wordnet.ensure_loaded()
        print("WordNet loaded successfully from configured path.")
    except LookupError:
        print(f"WARNING: Could not load WordNet from '{nltk_data_dir}' immediately after adding path.")
else:
    # Log a warning if the directory isn't found (helps debugging)
    print(f"WARNING: NLTK data directory not found at '{nltk_data_dir}'. Downloads might fail. Current NLTK paths: {nltk.data.path}")
# --- Tell NLTK where to find the data --- END ---


# --- Configuration & Constants ---
MODEL_FILENAME = 'hate_speech_model.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'
SENTIMENT_LABELS = {0: 'Not Hate', 1: 'Hate'}
UPLOAD_FOLDER = 'uploads' # Still good practice, though not saving locally on Render
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
ASSUMED_TEXT_COLUMN = 'Content'

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', b'_5#y2L"F4Q8z\n\xec]!') # Use env var or default

# --- NLTK Data Check (Should now work) ---
try:
    # Test loading stopwords and lemmatizing
    test_stopwords = stopwords.words('english')
    WordNetLemmatizer().lemmatize('tests')
    print(f"NLTK check successful: Found {len(test_stopwords)} stopwords.")
except LookupError as e:
    print("*" * 60)
    print(f"ERROR: Required NLTK data lookup failed: {e}")
    print(f"NLTK is searching in these paths: {nltk.data.path}")
    print("Check Render build logs: Did 'python -m nltk.downloader -d ./nltk_data ...' run successfully?")
    print("Check file system: Does the 'nltk_data' directory exist relative to 'app.py' with contents?")
    print("*" * 60)
    # Optionally raise an exception or exit if NLTK data is absolutely critical
    # raise RuntimeError("NLTK data failed to load, cannot continue.")


# --- Load Model and Vectorizer ---
model = None
vectorizer = None
model_classes = None
try:
    model_path = os.path.join(script_dir, MODEL_FILENAME)
    vectorizer_path = os.path.join(script_dir, VECTORIZER_FILENAME)
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        # Store model classes for robust probability indexing
        if hasattr(model, 'classes_'):
            model_classes = list(model.classes_)
            print(f"Model and vectorizer loaded successfully. Model classes: {model_classes}")
        else:
            print("Model loaded, but does not have 'classes_' attribute. Assuming binary [0, 1].")
            model_classes = [0, 1]
    else:
         print(f"ERROR: Model ('{model_path}') or Vectorizer ('{vectorizer_path}') file not found.")
         # Consider raising error if critical: raise FileNotFoundError(...)
except Exception as e:
    print(f"FATAL: Error loading model/vectorizer: {e}")
    # Optionally re-raise or handle gracefully
    # raise


# --- Text Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
try:
    # This should now succeed because NLTK knows where to look
    stop_words_set = set(stopwords.words('english'))
    print("Stopwords loaded successfully for clean_text function.")
except LookupError:
    print("WARNING: Stopwords list lookup failed during initialization! Preprocessing will not remove stopwords.")
    stop_words_set = set() # Fallback: proceed without stopword removal

def clean_text(text):
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Use the globally loaded stop_words_set
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

    # Check if model/vectorizer loaded correctly
    if model is None or vectorizer is None:
        error_message = "Model or Vectorizer is not available. Cannot perform analysis. Please check server logs."
        # Return status code 503 Service Unavailable might be appropriate
        return render_template('index.html', error_message=error_message), 503

    # --- BRANCH 1: File Upload Processing ---
    if file and file.filename != '':
        if allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                print(f"Processing uploaded file: {filename}")

                # Read file into pandas DataFrame
                if filename.lower().endswith('.csv'):
                     try:
                         # Use the file stream directly
                         df = pd.read_csv(file)
                     except UnicodeDecodeError:
                         file.seek(0) # Reset stream pointer
                         print("UTF-8 failed for CSV, trying latin-1")
                         df = pd.read_csv(file, encoding='latin-1')
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    # Use the file stream directly
                    df = pd.read_excel(file, engine='openpyxl')
                else:
                     error_message = "Unsupported file format."
                     return render_template('index.html', error_message=error_message)

                if ASSUMED_TEXT_COLUMN not in df.columns:
                    error_message = f"File processed, but required column '{ASSUMED_TEXT_COLUMN}' not found. Columns found: {', '.join(df.columns)}"
                    return render_template('index.html', error_message=error_message)

                results = []
                print(f"Found {len(df)} rows to process in the file.")
                for index, row in df.iterrows():
                    original_text = row[ASSUMED_TEXT_COLUMN]
                    if pd.isna(original_text) or not isinstance(original_text, str) or original_text.strip() == '':
                         pred_label = "Skipped (Invalid/Empty Input)"
                         prob_not_hate = None
                         prob_hate = None
                         cleaned_for_output = ""
                    else:
                        cleaned = clean_text(original_text)
                        vectorized = vectorizer.transform([cleaned])
                        prediction = model.predict(vectorized)[0]
                        proba = model.predict_proba(vectorized)[0]
                        pred_label = SENTIMENT_LABELS.get(prediction, f"Unknown Label ({prediction})") # More informative fallback

                        prob_not_hate = None
                        prob_hate = None
                        if model_classes and 0 in model_classes:
                            prob_not_hate = proba[model_classes.index(0)]
                        if model_classes and 1 in model_classes:
                             prob_hate = proba[model_classes.index(1)]

                        cleaned_for_output = cleaned

                    results.append({
                        'Original_Text': original_text,
                        'Cleaned_Text': cleaned_for_output,
                        'Prediction': pred_label,
                        'Confidence_Not_Hate': prob_not_hate,
                        'Confidence_Hate': prob_hate
                    })

                print("Finished processing file rows.")
                results_df = pd.DataFrame(results)
                output_buffer = io.BytesIO()
                results_df.to_csv(output_buffer, index=False, encoding='utf-8', float_format='%.4f')
                output_buffer.seek(0)

                print("Prepared results CSV for download.")
                return send_file(
                    output_buffer,
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name='hate_analysis_results.csv'
                )

            except Exception as e:
                print(f"Error processing file '{filename}': {e}")
                # import traceback # Use for detailed debugging if needed
                # print(traceback.format_exc())
                error_message = f"An error occurred processing the file: {e}"
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
                 prediction_text = SENTIMENT_LABELS.get(prediction, f"Unknown Classification ({prediction})") # More informative

                 probabilities_dict = {}
                 if model_classes and 0 in model_classes:
                     prob_0 = predicted_proba[model_classes.index(0)]
                     probabilities_dict['Not Hate'] = f"{prob_0:.1%}"
                 if model_classes and 1 in model_classes:
                      prob_1 = predicted_proba[model_classes.index(1)]
                      probabilities_dict['Hate'] = f"{prob_1:.1%}"

            else:
                error_message = "Prediction failed to return a valid result."

        except Exception as e:
            print(f"Error during single text prediction: {e}")
            # import traceback
            # print(traceback.format_exc())
            error_message = f"An error occurred during analysis: {e}"

        return render_template('index.html',
                               prediction_text=prediction_text,
                               probabilities=probabilities_dict,
                               submitted_text=submitted_text,
                               error_message=error_message,
                               info_message=None)

    # --- BRANCH 3: No Input Provided ---
    else:
        error_message = "Please enter text in the text area OR upload a file."
        return render_template('index.html', error_message=error_message)


# --- Run the App ---
if __name__ == '__main__':
    # Final check before starting server
    if model and vectorizer:
        # Determine port - Render provides PORT env var
        port = int(os.environ.get('PORT', 5000)) # Default to 5000 if not set
        # Use host='0.0.0.0' to be accessible externally (required by Render)
        # debug=False is crucial for production (set by default if FLASK_DEBUG not set)
        # Use Gunicorn in production, this block is mainly for local testing via `python app.py`
        print(f"Starting Flask server locally on host 0.0.0.0 port {port}...")
        app.run(host='0.0.0.0', port=port)
    else:
        print("*"*60)
        print("FATAL: Flask app cannot start - model or vectorizer failed to load.")
        print("Check logs above for loading errors.")
        print("*"*60)