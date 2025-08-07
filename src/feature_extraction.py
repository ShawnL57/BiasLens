# feature_extraction.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Step 2: Feature Extraction Only
# 1️⃣ Load combined_data.csv
data = pd.read_csv('processed/combined_data.csv')
X = data['text']  # Features: article text
y = data['label']  # Labels: left, center, right

# 2️⃣ Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# 3️⃣ Save feature matrix and labels as .pkl files
import joblib
joblib.dump(X_vectorized, 'processed/X_vectorized.pkl')
joblib.dump(y, 'processed/y_labels.pkl')
joblib.dump(vectorizer, "processed/vectorizer.pkl")
print("Feature extraction complete. TF-IDF matrix and labels saved.")
