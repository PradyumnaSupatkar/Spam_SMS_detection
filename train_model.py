import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load only this dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Use only the required columns from this dataset
df = df[["v1", "v2"]]

# Rename columns
df.columns = ["label", "message"]

# Convert labels: ham = 0, spam = 1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Clean SMS messages
df["message"] = df["message"].apply(clean_text)

# Remove missing values if any
df = df.dropna()

X = df["message"]
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Test model
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Spam SMS Detection Model")
print("------------------------")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved as model.pkl")
print("Vectorizer saved as vectorizer.pkl")