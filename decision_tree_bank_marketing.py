import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def download_and_extract_bank_data(download_url: str, target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    archive_path = os.path.join(target_dir, "bank-additional.zip")

    if not os.path.exists(archive_path):
        print("Downloading Bank Marketing dataset...")
        urllib.request.urlretrieve(download_url, archive_path)
        print("Download complete.")
    else:
        print("Archive already exists. Skipping download.")

    with zipfile.ZipFile(archive_path, "r") as z:
        z.extractall(target_dir)

    extracted_csv = os.path.join(target_dir, "bank-additional", "bank-additional-full.csv")
    if not os.path.exists(extracted_csv):
        raise FileNotFoundError(f"Expected file not found: {extracted_csv}")
    return extracted_csv


def load_bank_data(csv_path: str) -> pd.DataFrame:
    print(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_and_split(df: pd.DataFrame):
    target_column = "y"
    X = df.drop(columns=[target_column])
    y = (df[target_column] == "yes").astype(int)

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    return preprocessor, X_train, X_test, y_train, y_test


def build_and_evaluate(preprocessor, X_train, X_test, y_train, y_test):
    print("Training decision tree classifier...")
    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out()

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42, max_depth=6))
    ])

    print("Training decision tree classifier...")
    clf.fit(X_train, y_train)
    print("Training complete.")

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["no", "yes"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    tree_clf = clf.named_steps["classifier"]
    plt.figure(figsize=(18, 12))
    plot_tree(
        tree_clf,
        feature_names=feature_names,
        class_names=["no", "yes"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    tree_image_path = os.path.join(os.path.dirname(__file__), "decision_tree.png")
    plt.savefig(tree_image_path, bbox_inches="tight")
    plt.close()
    print(f"Decision tree image saved to: {tree_image_path}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no", "yes"])
    disp.plot(cmap=plt.cm.Blues)
    cm_image_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
    plt.savefig(cm_image_path, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix image saved to: {cm_image_path}")

    return clf


def main():
    download_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    )
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    csv_path = download_and_extract_bank_data(download_url, data_dir)
    df = load_bank_data(csv_path)
    preprocessor, X_train, X_test, y_train, y_test = preprocess_and_split(df)
    build_and_evaluate(preprocessor, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
plt.show()