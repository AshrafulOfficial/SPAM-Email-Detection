import os
import joblib

# ─── Path ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_rf_model.pkl")


# ─── Prediction Function ──────────────────────────────────────────────────────
def predict_email(email_text, model):
    prediction  = model.predict([email_text])[0]
    probability = model.predict_proba([email_text])[0]

    not_spam_probability = probability[0] * 100
    spam_probability     = probability[1] * 100

    result = "Spam Email" if prediction == 1 else "Not Spam / Ham Email"
    return result, spam_probability, not_spam_probability


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    model = joblib.load(MODEL_PATH)

    print("Spam Email Detector — Random Forest")
    print("Type 'exit' to stop.\n")

    while True:
        email_text = input("Enter email text: ")

        if email_text.lower() == "exit":
            print("Program closed.")
            break

        result, spam_prob, ham_prob = predict_email(email_text, model)

        print("\nPrediction        :", result)
        print(f"Spam probability  : {spam_prob:.2f}%")
        print(f"Not spam probability: {ham_prob:.2f}%\n")


if __name__ == "__main__":
    main()
