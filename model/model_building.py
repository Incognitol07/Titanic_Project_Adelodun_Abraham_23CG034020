import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os


def load_data():
    url = (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    print(f"Downloading data from {url}...")
    data = pd.read_csv(url)
    return data


def preprocess_and_train(data):
    # Selected features
    selected_features = ["Pclass", "Sex", "Age", "SibSp", "Fare"]
    target = "Survived"

    X = data[selected_features]
    y = data[target]

    # Preprocessing
    numeric_features = ["Age", "Fare", "SibSp", "Pclass"]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_features = ["Sex"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    print("Training Random Forest Model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return pipeline


if __name__ == "__main__":
    if not os.path.exists("model"):
        os.makedirs("model")

    df = load_data()
    model = preprocess_and_train(df)

    save_path = "model/titanic_survival_model.pkl"
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")
