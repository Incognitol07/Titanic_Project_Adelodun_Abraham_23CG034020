import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load Model
MODEL_PATH = "model/titanic_survival_model.pkl"
model = None


def load_titanic_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        print("Model file not found. Ensure model is trained.")


load_titanic_model()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    if request.method == "POST":
        if not model:
            load_titanic_model()

        if model:
            try:
                # Features: Pclass, Sex, Age, SibSp, Fare
                pclass = int(request.form["pclass"])
                sex = request.form["sex"]
                age = float(request.form["age"])
                sibsp = int(request.form["sibsp"])
                fare = float(request.form["fare"])

                # Create DataFrame
                input_df = pd.DataFrame(
                    [
                        {
                            "Pclass": pclass,
                            "Sex": sex,
                            "Age": age,
                            "SibSp": sibsp,
                            "Fare": fare,
                        }
                    ]
                )

                # Predict (0 = Died, 1 = Survived)
                pred = model.predict(input_df)[0]

                if pred == 1:
                    prediction_text = "Survived"
                else:
                    prediction_text = "Did Not Survive"

            except Exception as e:
                prediction_text = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
