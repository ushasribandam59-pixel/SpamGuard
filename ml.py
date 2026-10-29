"""SpamGuard machine-learning core.

Same pipeline as the original project (SMS Spam Collection -> cleaning ->
stratified 80/20 split -> TF-IDF(3000) -> Multinomial Naive Bayes), wrapped in
a class so the web app can reuse it, plus an explanation method that reports
which words pushed the score towards spam or ham using the model's *learned*
log-probabilities (no hard-coded results).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

log = logging.getLogger("spamguard.ml")


class SpamModel:
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self._train()

    # ------------------------------------------------------------------ train
    def _train(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.csv_path}. Place spam.csv "
                "(columns v1=label, v2=message) next to app.py.")
        data = pd.read_csv(self.csv_path, encoding="latin-1")
        data = data[["v1", "v2"]].copy()
        data.columns = ["label", "message"]
        data = data.dropna().drop_duplicates()
        data["label"] = data["label"].astype(str).str.strip().str.lower()
        data["message"] = data["message"].astype(str)
        data = data[data["label"].isin(["ham", "spam"])]
        self.data = data

        x_train_text, x_test_text, y_train, y_test = train_test_split(
            data["message"], data["label"], test_size=0.20,
            random_state=42, stratify=data["label"])
        self.vectorizer = TfidfVectorizer(max_features=3000)
        x_train = self.vectorizer.fit_transform(x_train_text)
        x_test = self.vectorizer.transform(x_test_text)
        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)

        classes = list(self.model.classes_)
        self._spam_idx = classes.index("spam")
        self._ham_idx = classes.index("ham")
        self._delta = (self.model.feature_log_prob_[self._spam_idx]
                       - self.model.feature_log_prob_[self._ham_idx])
        self._vocab = np.array(self.vectorizer.get_feature_names_out())

        cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
        report = classification_report(
            y_test, y_pred, labels=["ham", "spam"], target_names=["Ham", "Spam"],
            output_dict=True, zero_division=0)
        self.metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "precision": round(precision_score(y_test, y_pred, pos_label="spam", zero_division=0) * 100, 2),
            "recall": round(recall_score(y_test, y_pred, pos_label="spam", zero_division=0) * 100, 2),
            "f1": round(f1_score(y_test, y_pred, pos_label="spam", zero_division=0) * 100, 2),
            "test_total": int(len(y_test)),
            "train_total": int(len(y_train)),
            "ham_count": int((data["label"] == "ham").sum()),
            "spam_count": int((data["label"] == "spam").sum()),
            "matrix": {"tn": int(cm[0][0]), "fp": int(cm[0][1]),
                       "fn": int(cm[1][0]), "tp": int(cm[1][1])},
            "ham_report": {k: float(v) for k, v in report["Ham"].items()},
            "spam_report": {k: float(v) for k, v in report["Spam"].items()},
            "features": int(len(self._vocab)),
        }
        log.info("Model trained: accuracy=%.2f%% on %d test messages",
                 self.metrics["accuracy"], self.metrics["test_total"])

    # ---------------------------------------------------------------- predict
    def predict(self, text: str) -> dict:
        vec = self.vectorizer.transform([text])
        probs = self.model.predict_proba(vec)[0]
        spam_score = float(probs[self._spam_idx] * 100)
        prediction = self.model.classes_[int(np.argmax(probs))]

        # Per-word contribution to the spam-vs-ham log-odds (learned weights x tf-idf).
        idx = vec.indices
        contrib = vec.data * self._delta[idx]
        total = float(np.abs(contrib).sum()) or 1.0
        terms = [{"word": str(self._vocab[i]), "impact": round(float(abs(c)) / total * 100, 1),
                  "direction": "spam" if c > 0 else "ham"}
                 for i, c in zip(idx, contrib)]
        spam_terms = sorted((t for t in terms if t["direction"] == "spam"),
                            key=lambda t: -t["impact"])[:8]
        ham_terms = sorted((t for t in terms if t["direction"] == "ham"),
                           key=lambda t: -t["impact"])[:8]
        return {
            "prediction": str(prediction),
            "spam_score": round(spam_score, 2),
            "spam_terms": spam_terms,
            "ham_terms": ham_terms,
            "recognized_terms": int(len(idx)),
        }
