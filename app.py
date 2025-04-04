# File: app.py (Updated for "Hate" / "Not Hate" display)

import io
import os
import re
import string

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from werkzeug.utils import secure_filename
import numpy as np # Import numpy for checking model classes

# --- Configuration & Constants ---
MODEL_FILENAME = 'hate_speech_model.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'
# *** UPDATED LABELS ***
SENTIMENT_LABELS = {0: 'Not Hate', 1: 'Hate'} # Changed labels
UPLOAD_FOLDER = 'uploads' # Although not used for saving, good practice to define
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
ASSUMED_TEXT_COLUMN = 'Content' # Column name expected in uploaded files

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # Needed for flash messages or session if used

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
    # Consider exiting if NLTK data is critical and not found
    # exit()


# --- Load Model and Vectorizer ---
model = None
vectorizer = None
model_classes = None
try:
    if os.path.exists(MODEL_FILENAME) and os.path.exists(VECTORIZER_FILENAME):
        model = joblib.load(MODEL_FILENAME)
        vectorizer = joblib.load(VECTORIZER_FILENAME)
        # Store model classes for robust probability indexing
        if hasattr(model, 'classes_'):
            model_classes = list(model.classes_) # Convert to list for easy indexing
            print(f"Model and vectorizer loaded successfully. Model classes: {model_classes}")
        else:
            print("Model loaded, but does not have 'classes_' attribute. Assuming binary [0, 1] output.")
            model_classes = [0, 1] # Default assumption if attribute missing

    else:
         print(f"ERROR: Model ('{MODEL_FILENAME}') or Vectorizer ('{VECTORIZER_FILENAME}') file not found.")
         print("Please ensure the model has been trained using 'train_model.py' and the .pkl files are in the same directory.")
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")


# --- Text Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
try:
    stop_words_set = set(stopwords.words('english'))
except LookupError:
    print("Stopwords list not found (NLTK data issue?), proceeding without stopword removal.")
    stop_words_set = set()

def clean_text(text):
    if not isinstance(text, str): text = str(text) # Ensure input is string
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    # Lemmatize and remove stopwords
    cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words_set])
    return cleaned_text

# --- Helper Function for File Type Check ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def home():
    # Render the main page
    return render_template('index.html', prediction_text='', probabilities=None, submitted_text='', error_message=None, info_message=None)

@app.route('/predict', methods=['POST'])
def predict():
    prediction_text = ''
    probabilities_dict = None
    error_message = None
    info_message = None
    submitted_text = request.form.get('text_input', '').strip()
    file = request.files.get('file_upload') # Get file from form

    # Check if model/vectorizer loaded correctly
    if model is None or vectorizer is None:
        error_message = "Model or Vectorizer failed to load. Cannot perform analysis. Please check server logs."
        return render_template('index.html', error_message=error_message)

    # --- BRANCH 1: File Upload Processing ---
    if file and file.filename != '':
        if allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename) # Sanitize filename
                print(f"Processing uploaded file: {filename}")

                # Read file into pandas DataFrame
                if filename.lower().endswith('.csv'):
                     # Try reading with common encodings
                     try:
                         df = pd.read_csv(file)
                     except UnicodeDecodeError:
                         file.seek(0) # Reset file pointer
                         print("UTF-8 failed for CSV, trying latin-1")
                         df = pd.read_csv(file, encoding='latin-1')
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    # Let pandas handle Excel reading with openpyxl
                    df = pd.read_excel(file, engine='openpyxl')
                else:
                     # Should not happen due to allowed_file check, but good practice
                     error_message = "Unsupported file format."
                     return render_template('index.html', error_message=error_message)


                # Check if the required text column exists
                if ASSUMED_TEXT_COLUMN not in df.columns:
                    error_message = f"File processed, but required column '{ASSUMED_TEXT_COLUMN}' not found. Columns found: {', '.join(df.columns)}"
                    return render_template('index.html', error_message=error_message)

                results = []
                print(f"Found {len(df)} rows to process in the file.")
                # Iterate through rows, predict, and store results
                for index, row in df.iterrows():
                    original_text = row[ASSUMED_TEXT_COLUMN]
                    # Handle missing or non-string data in the text column
                    if pd.isna(original_text) or not isinstance(original_text, str) or original_text.strip() == '':
                         pred_label = "Skipped (Invalid/Empty Input)"
                         prob_not_hate = None
                         prob_hate = None
                         cleaned_for_output = "" # Add cleaned text column
                    else:
                        cleaned = clean_text(original_text)
                        vectorized = vectorizer.transform([cleaned])
                        prediction = model.predict(vectorized)[0]
                        proba = model.predict_proba(vectorized)[0]
                        pred_label = SENTIMENT_LABELS.get(prediction, "Unknown Label") # Map 0/1 to "Not Hate"/"Hate"

                        # *** Robust Probability assignment using model_classes ***
                        prob_not_hate = None
                        prob_hate = None
                        if 0 in model_classes:
                            prob_not_hate = proba[model_classes.index(0)]
                        if 1 in model_classes:
                             prob_hate = proba[model_classes.index(1)]

                        cleaned_for_output = cleaned # Save cleaned text

                    results.append({
                        'Original_Text': original_text,
                        'Cleaned_Text': cleaned_for_output, # Include cleaned text
                        'Prediction': pred_label,
                        # *** UPDATED Probability keys for CSV output ***
                        'Confidence_Not_Hate': prob_not_hate,
                        'Confidence_Hate': prob_hate
                    })

                print("Finished processing file rows.")
                results_df = pd.DataFrame(results)

                # Prepare CSV for download
                output_buffer = io.BytesIO()
                # Format probabilities nicely in the CSV
                results_df.to_csv(output_buffer, index=False, encoding='utf-8', float_format='%.4f')
                output_buffer.seek(0) # Rewind buffer

                print("Prepared results CSV for download.")
                # Send the CSV file back to the user
                return send_file(
                    output_buffer,
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name='hate_analysis_results.csv' # Changed filename
                )

            except Exception as e:
                print(f"Error processing file: {e}")
                error_message = f"An error occurred while processing the file: {e}"
                # Log the full traceback here if needed for debugging
                # import traceback
                # print(traceback.format_exc())
                return render_template('index.html', error_message=error_message)

        else:
            # File type not allowed
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
                 # Map prediction to display label
                 prediction_text = SENTIMENT_LABELS.get(prediction, "Unknown Classification")

                 # *** UPDATED Probability Dictionary using model_classes ***
                 probabilities_dict = {}
                 if 0 in model_classes:
                     prob_0 = predicted_proba[model_classes.index(0)]
                     probabilities_dict['Not Hate'] = f"{prob_0:.1%}" # Format as percentage
                 if 1 in model_classes:
                      prob_1 = predicted_proba[model_classes.index(1)]
                      probabilities_dict['Hate'] = f"{prob_1:.1%}" # Format as percentage

            else:
                # Handle case where prediction fails unexpectedly
                error_message = "Prediction failed to return a valid result."

        except Exception as e:
            print(f"Error during single text prediction: {e}")
            error_message = f"An error occurred during analysis: {e}"
             # Log the full traceback here if needed for debugging
             # import traceback
             # print(traceback.format_exc())

        # Render the page with results (or error) for single text input
        return render_template('index.html',
                               prediction_text=prediction_text,
                               probabilities=probabilities_dict,
                               submitted_text=submitted_text, # Keep user's text in box
                               error_message=error_message,
                               info_message=None)

    # --- BRANCH 3: No Input Provided ---
    else:
        # Neither text nor file was submitted
        error_message = "Please enter text in the text area OR upload a file."
        return render_template('index.html', error_message=error_message)


# --- Run the App ---
if __name__ == '__main__':
    # Ensure model and vectorizer are loaded before starting
    if model and vectorizer:
        print("Starting Flask server...")
        # Use host='0.0.0.0' to make it accessible on the network
        # debug=True is useful for development, but should be False in production
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("*"*60)
        print("ERROR: Flask app cannot start because the model or vectorizer failed to load.")
        print("Please check the file paths and ensure the model files exist.")
        print("Run 'python train_model.py' first if you haven't already.")
        print("*"*60)