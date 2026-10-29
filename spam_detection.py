
# ==========================================================
# EMAIL SPAM DETECTION USING NAIVE BAYES
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ==========================================================
# STEP 1: DATASET LOADING
# ==========================================================

print("\n" + "=" * 20)
print("STEP 1: DATASET LOADING")
print("=" * 50)

data = pd.read_csv("spam.csv", encoding="latin-1")

print("Original dataset shape:", data.shape)
print("\nFirst 5 rows of original dataset:")
print(data.head())


# ==========================================================
# STEP 2: SELECTING REQUIRED COLUMNS
# ==========================================================

print("\n" + "=" * 50)
print("STEP 2: SELECTING REQUIRED COLUMNS")
print("=" * 50)

data = data[["v1", "v2"]].copy()
data.columns = ["label", "message"]

print("Selected columns: label, message")
print("\nFirst 5 selected rows:")
print(data.head())


# ==========================================================
# STEP 3: DATA PREPROCESSING
# ==========================================================

print("\n" + "=" * 50)
print("STEP 3: DATA PREPROCESSING")
print("=" * 50)

print("\nMissing values before cleaning:")
print(data.isnull().sum())

print("\nDuplicate rows before cleaning:")
print(data.duplicated().sum())

data = data.drop_duplicates().copy()

print("\nRows after removing duplicates:", len(data))
print("Remaining duplicate rows:", data.duplicated().sum())

print("\nFirst 5 messages after preprocessing:")
print(data.head())


# ==========================================================
# STEP 4: MESSAGE COUNT ANALYSIS
# ==========================================================

print("\n" + "=" * 50)
print("STEP 4: MESSAGE COUNT ANALYSIS")
print("=" * 50)

print("\nNumber of messages in each category:")
print(data["label"].value_counts())

data["label"].value_counts().plot(
    kind="bar",
    title="Spam vs Ham Message Counts",
    xlabel="Message Category",
    ylabel="Number of Messages"
)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ==========================================================
# STEP 5: SEPARATING INPUT AND OUTPUT
# ==========================================================

print("\n" + "=" * 50)
print("STEP 5: SEPARATING INPUT AND OUTPUT")
print("=" * 50)

X = data["message"].astype(str)
y = data["label"]

print("Input: Message text")
print("Output: Spam or Ham")

print("\nFirst message:")
print(X.iloc[0])

print("First label:", y.iloc[0])


# ==========================================================
# STEP 6: TRAINING AND TESTING SPLIT
# ==========================================================

print("\n" + "=" * 50)
print("STEP 6: TRAINING AND TESTING SPLIT")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))


# ==========================================================
# STEP 7: FEATURE EXTRACTION USING TF-IDF
# ==========================================================

print("\n" + "=" * 50)
print("STEP 7: FEATURE EXTRACTION USING TF-IDF")
print("=" * 50)

tfidf = TfidfVectorizer(max_features=3000)

# Learn vocabulary from training data only
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform test data using the same vocabulary
X_test_tfidf = tfidf.transform(X_test)

print("Maximum features:", 3000)
print("Training feature matrix:", X_train_tfidf.shape)
print("Testing feature matrix:", X_test_tfidf.shape)

print("\nFirst 20 extracted words:")
print(tfidf.get_feature_names_out()[:20])


# ==========================================================
# STEP 8: NAIVE BAYES MODEL TRAINING
# ==========================================================

print("\n" + "=" * 50)
print("STEP 8: NAIVE BAYES MODEL TRAINING")
print("=" * 50)

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

print("Naive Bayes model trained successfully!")


# ==========================================================
# STEP 9: PREDICTING TEST MESSAGES
# ==========================================================

print("\n" + "=" * 50)
print("STEP 9: PREDICTING TEST MESSAGES")
print("=" * 50)

y_pred = model.predict(X_test_tfidf)

print("Predictions completed successfully!")

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nFirst 10 predictions:")
print(results.head(10).to_string(index=False))


# ==========================================================
# STEP 10: MODEL EVALUATION
# ==========================================================

print("\n" + "=" * 50)
print("STEP 10: MODEL EVALUATION")
print("=" * 50)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="spam")
recall = recall_score(y_test, y_pred, pos_label="spam")
f1 = f1_score(y_test, y_pred, pos_label="spam")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================================
# STEP 11: CONFUSION MATRIX
# ==========================================================

print("\n" + "=" * 50)
print("STEP 11: CONFUSION MATRIX")
print("=" * 50)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
)

print("Confusion Matrix:")
print(cm)

matrix_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

matrix_display.plot()
plt.title("Email Spam Detection - Confusion Matrix")
plt.tight_layout()
plt.show()


# ==========================================================
# STEP 12: CUSTOM MESSAGE TESTING
# ==========================================================

print("\n" + "=" * 50)
print("STEP 12: CUSTOM MESSAGE TESTING")
print("=" * 50)

message = input("Enter a message to check: ")

message_tfidf = tfidf.transform([message])

prediction = model.predict(message_tfidf)[0]

print("\nPrediction result:")

if prediction == "spam":
    print("Result: SPAM")
else:
    print("Result: NOT SPAM (HAM)")


# ==========================================================
# PROJECT COMPLETED
# ==========================================================

print("\n" + "=" * 50)
print("PROJECT EXECUTION COMPLETED")
print("=" * 50)
