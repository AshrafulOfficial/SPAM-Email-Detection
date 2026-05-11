import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "emails.csv")
FIGURE_DIR  = os.path.join(BASE_DIR, "figures")
MODEL_PATH  = os.path.join(BASE_DIR, "spam_rf_model.pkl")

TEXT_COL  = "text"
LABEL_COL = "spam"


# ─── 1. Directory ─────────────────────────────────────────────────────────────
def create_directory():
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)


# ─── 2. EDA ───────────────────────────────────────────────────────────────────
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

    df["email_length"] = df[TEXT_COL].astype(str).apply(len)
    df["word_count"]   = df[TEXT_COL].astype(str).apply(lambda x: len(x.split()))

    print("\nEmail Length Statistics:")
    print(df["email_length"].describe())
    print("\nWord Count Statistics:")
    print(df["word_count"].describe())

    return df


# ─── 3. Figure 1 — Confusion Matrix ──────────────────────────────────────────
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix — Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Not Spam", "Spam"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Not Spam", "Spam"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Figure saved: confusion_matrix.png")


# ─── 4. Figure 2 — Feature Importance ────────────────────────────────────────
def plot_feature_importance(pipeline):
    tfidf      = pipeline.named_steps["tfidf"]
    rf_model   = pipeline.named_steps["classifier"]
    feature_names = tfidf.get_feature_names_out()
    importances   = rf_model.feature_importances_

    # Top 20 most important words
    top_n   = 20
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores   = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#e74c3c" if s > top_scores.mean() else "#3498db" for s in top_scores]
    bars = ax.barh(range(top_n), top_scores[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance Score")
    ax.set_title("Top 20 Important Words — Random Forest", fontsize=13, fontweight="bold")
    ax.axvline(top_scores.mean(), color="gray", linestyle="--", linewidth=1, label="Mean")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("Figure saved: feature_importance.png")


# ─── 5. Figure 3 — ROC Curve ─────────────────────────────────────────────────
def plot_roc_curve(pipeline, X_test, y_test):
    y_proba    = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc    = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2,
            label=f"Random Forest (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Random Forest", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print(f"Figure saved: roc_curve.png  |  AUC = {roc_auc:.4f}")


# ─── 6. Figure 4 — Learning Curve ────────────────────────────────────────────
def plot_learning_curve(X_train, y_train):
    print("\nGenerating learning curve (this may take a moment)...")

    # Lightweight pipeline just for learning curve
    lc_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english", lowercase=True,
            max_features=5000, ngram_range=(1, 1)
        )),
        ("classifier", RandomForestClassifier(
            n_estimators=50, random_state=42, n_jobs=-1
        ))
    ])

    train_sizes, train_scores, val_scores = learning_curve(
        lc_pipeline, X_train, y_train,
        cv=3,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="accuracy",
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#e74c3c", label="Training Accuracy")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#e74c3c")
    ax.plot(train_sizes, val_mean, "o-", color="#2ecc71", label="Validation Accuracy")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#2ecc71")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve — Random Forest", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "learning_curve.png"), dpi=150)
    plt.close()
    print("Figure saved: learning_curve.png")


# ─── 7. Train ─────────────────────────────────────────────────────────────────
def train_model(df):
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    X  = df[TEXT_COL].astype(str)
    y  = df[LABEL_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n========== Train-Test Split ==========")
    print("Training data size:", len(X_train))
    print("Testing data size :", len(X_test))

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            max_features=10000,
            ngram_range=(1, 2)
        )),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\nTraining Random Forest model...")
    pipeline.fit(X_train, y_train)

    y_pred   = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)

    print("\n========== Model Evaluation ==========")
    print("Model Used : Random Forest Classifier")
    print("Accuracy   :", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"]))
    print("\nConfusion Matrix:")
    print(cm)

    # Generate all 4 figures
    plot_confusion_matrix(cm)
    plot_feature_importance(pipeline)
    plot_roc_curve(pipeline, X_test, y_test)
    plot_learning_curve(X_train, y_train)

    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print("\nModel saved as spam_rf_model.pkl")

    return accuracy


# ─── 8. Main ──────────────────────────────────────────────────────────────────
def main():
    create_directory()
    df = pd.read_csv(DATASET_PATH)
    df = perform_eda(df)
    accuracy = train_model(df)

    print("\n========== Final Summary ==========")
    print("Dataset    :", DATASET_PATH)
    print("Classifier : Random Forest")
    print("Train-Test : 80% training, 20% testing")
    print(f"Accuracy   : {accuracy * 100:.2f}%")
    print("Figures saved inside the figures folder.")


if __name__ == "__main__":
    main()
