
# Machine Learning Classification Dashboard

## 1. Problem Statement

This project compares five machine learning
classification algorithms using the Breast Cancer
Wisconsin (Diagnostic) dataset.

The objective is to evaluate their performance
using six classification metrics and demonstrate
their predictions through an interactive
Streamlit web application.

## 2. Dataset Description

- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Source: UCI Machine Learning Repository
- URL: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Total instances: 569
- Input features: 30
- Classification type: Binary
- Target: Diagnosis (Benign or Malignant)
- Training samples: 455
- Testing samples: 114
- Train-test split: 80:20
- Missing values: 0
- Duplicate rows: 0

The patient ID was excluded from model training.
Diagnosis was encoded as Benign = 0 and
Malignant = 1.

The data was split using stratified sampling
with random_state=42.

## 3. GitHub Repository Link

https://github.com/som101-tito/ML-Classification-Assignment-2

## 4. Machine Learning Models

The following models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbours
4. Gaussian Naive Bayes
5. Support Vector Machine

## 5. Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## 6. Model Performance Observations

### Logistic Regression

Achieved 96.49% accuracy and the highest
AUC score of 0.9960.

It correctly classified 110 of 114 test
samples, with three false negatives.

### Decision Tree

Achieved 90.35% accuracy.

It produced the lowest recall among the
five models and incorrectly classified
nine malignant cases as benign.

### K-Nearest Neighbours

Achieved 95.61% accuracy and an
F1 score of 0.9383.

It performed better than Decision Tree
but slightly below Logistic Regression.

### Gaussian Naive Bayes

Achieved 92.11% accuracy and
an AUC score of 0.9891.

It showed strong discrimination but
missed six malignant cases.

### Support Vector Machine

Achieved the highest accuracy of 97.37%,
precision of 1.0000, F1 score of 0.9630,
and MCC score of 0.9442.

It correctly classified 111 of 114
test samples.

## 7. Overall Winning Model

Support Vector Machine (SVM) was selected
as the overall winner based on its
highest F1 score.

It also achieved the best accuracy,
precision and MCC.

Logistic Regression achieved a slightly
higher AUC score.

Both SVM and Logistic Regression
achieved a recall of 0.9286.

## 8. Streamlit Application

The interactive application provides:

- CSV test dataset upload
- Selection among five trained models
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report

The application evaluates uploaded
labelled test data using the saved
machine learning models.

## 9. Running the Application

Install the required dependencies:

pip install -r requirements.txt

Launch the application:

streamlit run app.py

Upload test_data.csv and select a
classification model to view its results.

## 10. Streamlit Deployment Link

https://ml-classification-assignment-2-kt7xhtd5hbzwnmaycexyvu.streamlit.app

## 11. Execution Environment

The machine learning implementation,
evaluation and local Streamlit testing
were performed using BITS Virtual Lab.

A screenshot of the execution environment
will be included in the final
assignment submission PDF.
