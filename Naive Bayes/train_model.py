import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_PATH = "../emails.csv"
TEXT_COL = "text"
LABEL_COL = "spam"
FIGURE_DIR = "figures"


def create_directory():
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)


def perform_eda(df):
    print("\n========== Exploratory Data Analysis ==========")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nDataset Columns:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nLabel Distribution:")
    print(df[LABEL_COL].value_counts())

    # Email length and word count
    df["email_length"] = df[TEXT_COL].astype(str).apply(len)
    df["word_count"] = df[TEXT_COL].astype(str).apply(lambda x: len(x.split()))

    print("\nEmail Length Statistics:")
    print(df["email_length"].describe())

    print("\nWord Count Statistics:")
    print(df["word_count"].describe())

    # Label distribution chart
    plt.figure(figsize=(6, 4))
    df[LABEL_COL].value_counts().plot(kind="bar")
    plt.title("Spam and Not Spam Email Distribution")
    plt.xlabel("Class Label: 0 = Not Spam, 1 = Spam")
    plt.ylabel("Number of Emails")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/label_distribution.png")
    plt.close()

    # Email length distribution
    plt.figure(figsize=(7, 4))
    plt.hist(df["email_length"], bins=50)
    plt.title("Email Length Distribution")
    plt.xlabel("Email Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/email_length_distribution.png")
    plt.close()

    # Word count distribution
    plt.figure(figsize=(7, 4))
    plt.hist(df["word_count"], bins=50)
    plt.title("Word Count Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/word_count_distribution.png")
    plt.close()

    return df


def train_model(df):
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

    print("\n========== Train-Test Split ==========")
    print("Training data size:", len(X_train))
    print("Testing data size:", len(X_test))

    # Naive Bayes model pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            max_features=10000,
            ngram_range=(1, 2)
        )),
        ("classifier", MultinomialNB(alpha=0.1))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n========== Model Evaluation ==========")
    print("Model Used: Multinomial Naive Bayes")
    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Not Spam", "Spam"]
    ))

    print("\nConfusion Matrix:")
    print(cm)

    # Confusion matrix chart
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks([0, 1], ["Not Spam", "Spam"])
    plt.yticks([0, 1], ["Not Spam", "Spam"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/confusion_matrix.png")
    plt.close()

    # Save model
    joblib.dump(model, "spam_email_detector.pkl")

    print("\nModel saved successfully as spam_email_detector.pkl")

    return accuracy


def main():
    create_directory()

    df = pd.read_csv(DATASET_PATH)

    df = perform_eda(df)

    accuracy = train_model(df)

    print("\n========== Final Summary ==========")
    print("Dataset:", DATASET_PATH)
    print("Classifier: Multinomial Naive Bayes")
    print("Train-Test Split: 80% training, 20% testing")
    print(f"Final Accuracy: {accuracy * 100:.2f}%")
    print("EDA charts saved inside the figures folder.")


if __name__ == "__main__":
    main()