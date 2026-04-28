import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_PATH = "emails.csv"
TEXT_COL = "text"
LABEL_COL = "spam"


def main():
    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\nLabel distribution:")
    print(df[LABEL_COL].value_counts())

    # Clean dataset
    df = df[[TEXT_COL, LABEL_COL]].dropna()

    X = df[TEXT_COL].astype(str)
    y = df[LABEL_COL].astype(int)

    # 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Improved model
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            max_features=10000,
            ngram_range=(1, 2)
        )),
        ("classifier", LinearSVC())
    ])

    # Train model
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Not Spam", "Spam"]
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(model, "spam_email_detector.pkl")

    print("\nImproved model saved successfully as spam_email_detector.pkl")


if __name__ == "__main__":
    main()