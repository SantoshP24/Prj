# File: train_model.py

import os  # Import os module to check file existence
import re
import string

import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# --- NLTK Data Check/Download ---
# It's good practice to ensure NLTK data is available.
# Run the download commands manually in the PyCharm Terminal if needed:
# python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
try:
    stopwords.words('english')
    WordNetLemmatizer().lemmatize('test')
    print("NLTK data seems available.")
except LookupError:
    print("NLTK data ('stopwords', 'wordnet', 'omw-1.4') not found.")
    print("Please run the NLTK download commands in your PyCharm Terminal.")
    print("Example: python -c \"import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')\"")
    exit() # Exit if data is missing


# --- Configuration ---
DATASET_PATH = 'HateSpeechDataset.csv'  # Your dataset filename
TEXT_COLUMN = 'Content'       # Your text column name
LABEL_COLUMN = 'Label'             # Your label column name

label_mapping = None # Labels are assumed to be already numerical (0, 1)

# Names corresponding to numerical labels 0 and 1.
# Adjust if 0 is not Neutral or 1 is not Offensive/Hate in your data.
TARGET_CLASS_NAMES = ['Neutral', 'Offensive/Hate']

# --- Load Data ---
print(f"\nLoading data from: {DATASET_PATH}")
if not os.path.exists(DATASET_PATH):
     print(f"Error: Dataset file not found at '{DATASET_PATH}'.")
     print(f"Please make sure '{DATASET_PATH}' is in the same directory as train_model.py.")
     exit()

try:
    # Try detecting encoding issues, common with text data
    # Explicitly state header=0 (though it's the default)
    try:
        df = pd.read_csv(DATASET_PATH, header=0)
    except UnicodeDecodeError:
        print("UTF-8 decoding failed, trying latin-1...")
        df = pd.read_csv(DATASET_PATH, header=0, encoding='latin-1') # Or 'iso-8859-1'

    print(f"Dataset loaded successfully with {len(df)} rows.")

    # --- DEBUGGING: Print head and dtypes ---
    print("\n<< DEBUGGING INFO START >>")
    print("First 5 rows of loaded data (check if header is row 0):")
    print(df.head())
    print("\nData types of columns (check 'Label' type):")
    print(df.dtypes)
    print("<< DEBUGGING INFO END >>\n")
    # --- END DEBUGGING ---

    # Basic sanity checks
    if TEXT_COLUMN not in df.columns:
        if len(df) > 0 and TEXT_COLUMN in df.iloc[0].values:
             print(f"Warning: Column name '{TEXT_COLUMN}' found in the first data row. Header might not be parsed correctly.")
        raise ValueError(f"Text column '{TEXT_COLUMN}' not found in the dataset columns: {df.columns.tolist()}")
    if LABEL_COLUMN not in df.columns:
        if len(df) > 0 and LABEL_COLUMN in df.iloc[0].values:
             print(f"Warning: Column name '{LABEL_COLUMN}' found in the first data row. Header might not be parsed correctly.")
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found in the dataset columns: {df.columns.tolist()}")


    # Drop rows with missing text or labels (do this AFTER checking columns)
    initial_rows = len(df)
    df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)
    print(f"Rows before dropping NAs: {initial_rows}, Rows after dropping NAs: {len(df)}")
    if len(df) == 0:
         raise ValueError("No valid data remaining after dropping NAs.")

except ValueError as ve:
    print(f"Data validation error: {ve}")
    exit()
except Exception as e:
    print(f"Error loading or validating dataset: {e}")
    exit()


# --- Assign Target Variable ---
print("\nAttempting to convert label column to integer...")

# --->>> FILTERING STEP TO REMOVE INVALID 'Label' STRINGS or other non-numeric values <<<---
print(f"Filtering out rows where '{LABEL_COLUMN}' column is not '0' or '1'...")
initial_rows_before_filter = len(df)
# Keep only rows where the value in the LABEL_COLUMN IS '0' or '1' (as strings or numbers)
# Convert to string first to handle potential mixed types robustly
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)
valid_labels = ['0', '1']
df = df[df[LABEL_COLUMN].isin(valid_labels)] # Keep only rows where Label is '0' or '1'
rows_after_filter = len(df)
print(f"Rows before filter: {initial_rows_before_filter}, Rows after filtering: {rows_after_filter}")
removed_count = initial_rows_before_filter - rows_after_filter
if removed_count > 0:
    print(f"Removed {removed_count} rows containing invalid values in '{LABEL_COLUMN}'.")
if rows_after_filter == 0:
     print(f"Error: No data remaining after filtering out invalid values in '{LABEL_COLUMN}'.")
     exit()
# --->>> END OF FILTERING STEP <<<---

try:
    # Now convert the cleaned column to integer
    df['target'] = df[LABEL_COLUMN].astype(int)
    print("Label column converted successfully.")
    print("\nLabel distribution:")
    print(df['target'].value_counts())
except ValueError as e: # Catch specifically the ValueError for invalid literal
    print(f"--- ERROR ---")
    print(f"Error converting label column '{LABEL_COLUMN}' to integer: {e}")
    print("This happened AFTER filtering. There might be unexpected non-numeric values remaining.")
    print(f"\nUnique values remaining in '{LABEL_COLUMN}' column causing issues (first 10 unique):")
    try:
        problem_values = df[LABEL_COLUMN].unique()
        print(problem_values[:10])
    except Exception as e_inner:
        print(f"(Could not retrieve unique values: {e_inner})")
    exit()
except Exception as e: # Catch other potential errors during conversion
    print(f"--- ERROR ---")
    print(f"An unexpected error occurred converting '{LABEL_COLUMN}' to integer: {e}")
    exit()

# --- Text Preprocessing ---
print("\nPreprocessing text...")
lemmatizer = WordNetLemmatizer()
stop_words_list = stopwords.words('english')

def clean_text(text):
    # Handle potential non-string data
    if not isinstance(text, str):
        text = str(text)

    text = text.lower() # Ensure text is string and lowercase
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    # Tokenize, remove stopwords, lemmatize
    cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words_list])
    return cleaned_text

# Apply cleaning (can take time on large datasets)
df['cleaned_text'] = df[TEXT_COLUMN].apply(clean_text)
print("Text preprocessing complete.")


# --- Feature Engineering (TF-IDF) ---
print("\nApplying TF-IDF Vectorizer...")
# Adjust max_features based on performance/memory
vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(df['cleaned_text'])
y = df['target']
print(f"TF-IDF matrix shape: {X_tfidf.shape}")


# --- Train-Test Split ---
print("\nSplitting data into train and test sets...")
try:
    # Ensure there are enough samples in each class for stratification
    min_class_count = df['target'].value_counts().min()
    if min_class_count < 2: # Need at least 2 samples per class for split + stratify
         raise ValueError(f"The smallest class has only {min_class_count} samples, which is insufficient for train/test splitting with stratification. Need at least 2.")

    test_size = 0.2
    # Adjust test_size if one class has very few samples (e.g., 2 or 3)
    # Ensure at least 1 sample of each class is in the test set
    if min_class_count * test_size < 1:
        # Calculate minimum test_size to get at least 1 sample of the minority class
        required_test_size = 1.0 / min_class_count
        print(f"Warning: Adjusting test_size to {required_test_size:.3f} to ensure minority class is present in test set.")
        test_size = required_test_size

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y,
        test_size=test_size,     # Adjusted test size
        random_state=42,   # For reproducibility
        stratify=y         # IMPORTANT for imbalanced datasets
    )
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print("Train set label distribution:")
    print(y_train.value_counts())
    print("Test set label distribution:")
    print(y_test.value_counts())

except ValueError as e:
    print(f"Error during train/test split: {e}")
    print("This might happen if one class has too few samples for stratification, even after filtering.")
    print("Check the final label distribution carefully:")
    print(df['target'].value_counts())
    exit()


# --- Model Training (Logistic Regression with Class Weighting) ---
print("\nTraining Logistic Regression model...")
# Using class_weight='balanced' helps handle imbalanced data
model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    solver='liblinear', # Good solver for binary classification
    max_iter=1000      # Increase iterations for convergence
)
model.fit(X_train, y_train)
print("Model training complete.")


# --- Evaluation ---
print("\nEvaluating model...")
try:
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    # Ensure target_names matches the number of unique classes found in y_test/y_pred
    unique_labels = np.unique(np.concatenate((y_test, y_pred))) # Combine true and predicted labels to find all occurring labels
    unique_labels.sort() # Ensure order [0, 1]

    # Check if both 0 and 1 are present in the actual test labels or predictions
    if len(unique_labels) == 2 and unique_labels[0] == 0 and unique_labels[1] == 1:
         current_target_names = [TARGET_CLASS_NAMES[0], TARGET_CLASS_NAMES[1]]
         print(classification_report(y_test, y_pred, target_names=current_target_names))
    elif len(unique_labels) == 1:
         # Only one class predicted or present in y_test
         present_label = unique_labels[0]
         if present_label < len(TARGET_CLASS_NAMES):
             current_target_names = [TARGET_CLASS_NAMES[present_label]]
             print(f"Warning: Only one class ({TARGET_CLASS_NAMES[present_label]}) present in test results. Evaluation might be limited.")
             print(classification_report(y_test, y_pred, target_names=current_target_names, zero_division=0))
         else:
             print("Warning: Only one class present, but its index is out of bounds for TARGET_CLASS_NAMES.")
             print(classification_report(y_test, y_pred, zero_division=0))
    else:
        print("Warning: Unexpected unique labels found in test results. Using default numeric labels.")
        print(f"Unique labels found: {unique_labels}")
        print(classification_report(y_test, y_pred, zero_division=0))

except Exception as e:
    print(f"Error during evaluation: {e}")
    print("Trying evaluation without target names...")
    try:
        print(classification_report(y_test, y_pred, zero_division=0))
    except Exception as e_inner:
        print(f"Secondary evaluation attempt failed: {e_inner}")


# --- Save Model and Vectorizer ---
print("\nSaving model and vectorizer...")
try:
    joblib.dump(model, 'hate_speech_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Model and vectorizer saved successfully as 'hate_speech_model.pkl' and 'tfidf_vectorizer.pkl'")
except Exception as e:
    print(f"Error saving model/vectorizer: {e}")

print("\n--- Training script finished ---")