# Task-1
Iris Flower Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the uploaded dataset
file_path = '/mnt/data/Iris.csv'
iris_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(iris_data.head())
print(iris_data.info())

# Drop the 'Id' column and prepare features and target variable
X = iris_data.drop(columns=['Id', 'Species'])
y = iris_data['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)
print('Confusion Matrix:\n', conf_matrix)
