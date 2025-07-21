import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# load dataset
data_path = "data/iris.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError("Dataset not found in /data folder.")

df = pd.read_csv(data_path)

df.rename(columns={
    "SepalLengthCm": "sepal_length",
    "SepalWidthCm": "sepal_width",
    "PetalLengthCm": "petal_length",
    "PetalWidthCm": "petal_width",
    "Species": "species"
}, inplace=True)

if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)


# check data
if df.isnull().sum().any():
    print("Warning: Dataset contains missing values.")
    df = df.dropna()

# split features and target
X = df.drop("species", axis=1)
y = df["species"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# prediction and evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {round(acc * 100, 2)}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# save model
os.makedirs("model", exist_ok=True)
model_path = "model/iris_model.pkl"
joblib.dump(model, model_path)
print(f"\n✅ Trained model saved at: {model_path}")
