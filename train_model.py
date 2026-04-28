import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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

    # Keep required columns and remove missing values
    df = df[[TEXT_COL, LABEL_COL]].dropna()

    X = df[TEXT_COL].astype(str)
    y = df[LABEL_COL].astype(int)

    # Split dataset into training and testing parts
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Machine learning pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            max_features=5000
        )),
        ("classifier", MultinomialNB())
    ])

    # Train model
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Not Spam", "Spam"]
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save trained model
    joblib.dump(model, "spam_email_detector.pkl")

    print("\nModel saved successfully as spam_email_detector.pkl")


if __name__ == "__main__":
    main()