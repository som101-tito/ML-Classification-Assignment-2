
import streamlit as st

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")

st.write(
    "An interactive comparison of five machine learning "
    "classification algorithms using the Breast Cancer "
    "Wisconsin (Diagnostic) dataset."
)

st.divider()

st.subheader("Project Overview")

st.write("Dataset: Breast Cancer Wisconsin (Diagnostic)")
st.write("Total records: 569")
st.write("Input features: 30")
st.write("Classification type: Binary")

st.info(
    "This application compares Logistic Regression, "
    "Decision Tree, KNN, Gaussian Naive Bayes, and "
    "Support Vector Machine."
)

st.caption(
    "M.Tech Data Science & Engineering | "
    "Machine Learning Assignment 2"
)


# CSV Test Data Upload

import pandas as pd

st.divider()
st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test dataset (CSV format)",
    type=["csv"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    st.success("Test dataset uploaded successfully!")

    st.write("Uploaded dataset shape:", test_df.shape)

    st.write("Preview of uploaded data:")
    st.dataframe(test_df.head())

else:
    st.info("Please upload test_data.csv to continue.")


# Model Selection

st.divider()
st.subheader("Select Classification Model")

model_options = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Gaussian Naive Bayes",
    "Support Vector Machine"
]

selected_model = st.selectbox(
    "Choose a machine learning model",
    model_options
)

st.write("Selected model:", selected_model)



# Step 14F: Model prediction and evaluation

import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Map dropdown selections to saved model files
model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

if uploaded_file is not None:

    if "diagnosis" not in test_df.columns:
        st.error("The CSV must contain a diagnosis column.")

    else:
        # Prepare test features and actual labels
        X_uploaded = test_df.drop(columns=["diagnosis"])
        y_uploaded = test_df["diagnosis"].map({"B": 0, "M": 1})

        if y_uploaded.isna().any():
            st.error("Diagnosis must contain B or M labels.")

        else:
            model_path = (
                Path(__file__).resolve().parent
                / "model"
                / model_files[selected_model]
            )

            if not model_path.exists():
                st.error("Saved model file was not found.")

            else:
                # Load the selected trained model
                model = joblib.load(model_path)

                # Make predictions
                predictions = model.predict(X_uploaded)
                probabilities = model.predict_proba(X_uploaded)[:, 1]

                # Calculate evaluation metrics
                scores = {
                    "Accuracy": accuracy_score(y_uploaded, predictions),
                    "AUC": roc_auc_score(y_uploaded, probabilities),
                    "Precision": precision_score(y_uploaded, predictions),
                    "Recall": recall_score(y_uploaded, predictions),
                    "F1": f1_score(y_uploaded, predictions),
                    "MCC": matthews_corrcoef(y_uploaded, predictions)
                }

                st.divider()
                st.subheader("Model Evaluation Results")

                st.success(f"Evaluated: {selected_model}")

                # Show all six required metrics
                cols = st.columns(3)

                for i, (metric, value) in enumerate(scores.items()):
                    cols[i % 3].metric(metric, f"{value:.4f}")

                # Display confusion matrix
                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_uploaded,
                    predictions,
                    labels=[0, 1]
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Benign", "Actual Malignant"],
                    columns=["Predicted Benign", "Predicted Malignant"]
                )

                st.dataframe(cm_df)

                # Display classification report
                st.subheader("Classification Report")

                report = classification_report(
                    y_uploaded,
                    predictions,
                    labels=[0, 1],
                    target_names=["Benign", "Malignant"],
                    output_dict=True,
                    zero_division=0
                )

                st.dataframe(pd.DataFrame(report).transpose())

                st.caption(
                    "These results are calculated using the "
                    "uploaded labelled test dataset."
                )

