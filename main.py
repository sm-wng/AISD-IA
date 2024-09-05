import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
print("Loading datasets...")
train_data = pd.read_csv('/home/hwgroup/Desktop/AISD-interviewAssignment/train.csv')
test_data = pd.read_csv('/home/hwgroup/Desktop/AISD-interviewAssignment/test.csv')

# Identify categorical columns
categorical_cols = train_data.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical features
print("Applying one-hot encoding to categorical features...")
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# One-hot encode training data
encoded_train = encoder.fit_transform(train_data[categorical_cols])
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols))
train_data_encoded = pd.concat([train_data.drop(columns=categorical_cols), encoded_train_df], axis=1)

# One-hot encode test data
encoded_test = encoder.transform(test_data[categorical_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))
test_data_encoded = pd.concat([test_data.drop(columns=categorical_cols), encoded_test_df], axis=1)

# Separate features and target
X = train_data_encoded.drop(columns=['Exited'])
y = train_data_encoded['Exited']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
print("Training the Random Forest model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the validation set
print("Making predictions on the validation set...")
y_pred = model.predict(X_val)

# Evaluate the model
f1 = f1_score(y_val, y_pred)
print(f"F1 Score: {f1}")
conf_matrix = confusion_matrix(y_val, y_pred)

# Plot the confusion matrix
print("Plotting confusion matrix...")
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.show()

# Predict on the test data
print("Making predictions on the test data...")
test_predictions = model.predict(test_data_encoded)

# Save the predictions to a CSV file
output = pd.DataFrame({'Predicted_Exited': test_predictions})
output.to_csv('/home/hwgroup/Desktop/AISD-interviewAssignment/predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'.")
