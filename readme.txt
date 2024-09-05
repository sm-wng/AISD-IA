Overview

This project uses a Random Forest model to predict customer churn based on various attributes. The task involves preprocessing data, training a model, and evaluating its performance.
Files

    train.csv: Training data with features and target.
    test.csv: Test data with features only.
    predictions.csv: File with churn predictions for test data.
    main.py: Python script for data preprocessing, model training, and prediction.

Steps
1. Data Loading

Loads and displays the shapes and types of the training and test datasets.
2. Data Preprocessing

    Categorical Columns: Identified and one-hot encoded.
    Missing Values: Checked and handled.

3. Model Training

    Feature and Target Separation: Features and target variable separated from training data.
    Data Split: Training data split into training and validation sets.
    Model: Trained a Random Forest Classifier.

4. Model Evaluation

    Validation: Evaluated with F1 Score and confusion matrix.
    Confusion Matrix: Visualized to assess model performance.

5. Predictions

    Test Data: Predictions made for test data.
    Output: Predictions saved to predictions.csv.

Results

    F1 Score: 0.6030
    Confusion Matrix: included in file

Notes

    Ensure all necessary libraries (pandas, scikit-learn, matplotlib, seaborn) are installed.
    Paths in the script should be updated as per your directory structure.due to the confidentiality of the data, data is not hosted in this repository. Download data files elsewhere.
