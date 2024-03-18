import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('/content/heart.csv')

# Data Exploration
print("Shape of the dataset:", df.shape)
print("\nFirst few rows of the dataset:")
print(df.head())
print("\nSummary statistics of the dataset:")
print(df.describe())
print("\nInformation about the dataset:")
print(df.info())
print("\nUnique values in the 'target' column:")
print(df["target"].unique())
print("\nCorrelation with the target variable:")
print(df.corr()["target"].abs().sort_values(ascending=False))

# Preprocessing
X = df.drop("target", axis=1)
y = df["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with increased max_iter
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
accuracy_lr = round(accuracy_score(y_pred_lr, y_test) * 100, 2)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
accuracy_rf = round(accuracy_score(y_pred_rf, y_test) * 100, 2)

# Print accuracy scores

print("\nAccuracy of Logistic Regression after tuning: {}%".format(accuracy_lr))
print("Accuracy of Random Forest Classifier: {}%".format(accuracy_rf))
