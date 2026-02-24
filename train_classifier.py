import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ---------------- CONFIG ----------------

DATASET_PATH = "legal_ai_dataset_500_entries.csv"
MODEL_OUTPUT_PATH = "legal_outcome_classifier.joblib"

# ---------------- LOAD DATA ----------------

print("🔹 Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Basic validation
required_columns = {"text", "label"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Dataset must contain columns: {required_columns}")

# Clean labels
df["label"] = df["label"].str.strip().str.title()

print("\nLabel distribution:")
print(df["label"].value_counts())

# ---------------- EMBEDDINGS ----------------

print("\n🔹 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("🔹 Creating embeddings (this may take a moment)...")
X = embedder.encode(df["text"].tolist(), show_progress_bar=True)
y = df["label"]

# ---------------- TRAIN / TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- TRAIN CLASSIFIER ----------------

print("\n🔹 Training Legal Outcome Classifier...")
classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)
classifier.fit(X_train, y_train)

# ---------------- EVALUATION ----------------

print("\n📊 Evaluation Results:\n")
y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---------------- SAVE MODEL ----------------

joblib.dump(classifier, MODEL_OUTPUT_PATH)
print(f"\n✅ Trained model saved as: {MODEL_OUTPUT_PATH}")
