from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/titanic_survival_model.pkl")

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Collect input values from HTML form
        features = [
            int(request.form["Pclass"]),
            int(request.form["Sex"]),   # 0 or 1 (male/female)
            float(request.form["Age"]),
            int(request.form["SibSp"]),
            float(request.form["Fare"])
        ]
        final_input = np.array(features).reshape(1, -1)
        pred = model.predict(final_input)[0]
        prediction = "Survived" if pred == 1 else "Did Not Survive"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
