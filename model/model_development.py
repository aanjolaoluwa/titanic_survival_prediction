import pandas as pd

# Sample Titanic dataset
data = {
    "Pclass": [1, 3, 2, 3, 1, 3],
    "Sex": ["male", "female", "female", "male", "female", "male"],
    "Age": [38, 26, 35, 30, 28, 40],
    "SibSp": [1, 0, 0, 0, 1, 0],
    "Fare": [71.28, 7.25, 13.0, 8.05, 53.1, 8.46],
    "Embarked": ["C", "S", "S", "S", "C", "S"],
    "Survived": [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier  # Example algorithm
from sklearn.metrics import classification_report
import joblib

# Encode categorical features
le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])

le_emb = LabelEncoder()
df["Embarked"] = le_emb.fit_transform(df["Embarked"])

# Select features (any 5, excluding target)
X = df[["Pclass", "Sex", "Age", "SibSp", "Fare"]]  # Example 5 features
y = df["Survived"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model/titanic_survival_model.pkl")
print("Model saved!")
loaded_model = joblib.load("model/titanic_survival_model.pkl")
sample = X_test.iloc[0].values.reshape(1, -1)
print("Prediction:", loaded_model.predict(sample)[0])
