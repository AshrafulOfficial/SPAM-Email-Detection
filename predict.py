import joblib


MODEL_PATH = "spam_email_detector.pkl"


def predict_email(email_text, model):
    prediction = model.predict([email_text])[0]

    if prediction == 1:
        result = "Spam Email"
    else:
        result = "Not Spam / Ham Email"

    return result


def main():
    model = joblib.load(MODEL_PATH)

    print("Spam Email Detector")
    print("Type 'exit' to stop.\n")

    while True:
        email_text = input("Enter email text: ")

        if email_text.lower() == "exit":
            print("Program closed.")
            break

        result = predict_email(email_text, model)

        print("\nPrediction:", result, "\n")


if __name__ == "__main__":
    main()