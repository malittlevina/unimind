import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

class EmotionClassifier:
    def __init__(self, model_path='emotion/emotion_model.pkl', vectorizer_path='emotion/vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            self.model = LogisticRegression()
            self.vectorizer = TfidfVectorizer()
            self.train_default_model()

    def train_default_model(self):
        default_texts = [
            "I am so happy and excited",
            "This is so sad and depressing",
            "I feel angry and frustrated",
            "What a surprise! I didn't expect that",
            "I am afraid and nervous"
        ]
        default_labels = ["happy", "sad", "angry", "surprised", "fearful"]
        X = self.vectorizer.fit_transform(default_texts)
        self.model.fit(X, default_labels)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)

    def predict_emotion(self, text):
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

    def update_model(self, texts, labels):
        X_new = self.vectorizer.fit_transform(texts)
        self.model.fit(X_new, labels)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)

# Example usage
if __name__ == "__main__":
    ec = EmotionClassifier()
    print(ec.predict_emotion("I am very frustrated with this situation"))