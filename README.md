# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the placement dataset and remove unnecessary columns such as serial number and salary.
2. Convert all categorical attributes into numerical values using label encoding.
3.Split the dataset into training and testing sets using train–test split.
4. Train a Logistic Regression model using the training data. 5.Predict placement status and evaluate the model using accuracy and confusion matrix.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SAI PRAVEEN C
RegisterNumber:212225230239
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# --- Visual-kaga intha rendu library add pannuraen ---
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Previous_Score': [40, 50, 55, 60, 65, 70, 75, 80],
    'Internship': [0, 0, 1, 0, 1, 1, 1, 1],
    'Placement': [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Split into features and target
# ------------------------------
X = df[['Hours_Studied', 'Previous_Score', 'Internship']]
y = df['Placement']

# ------------------------------
# Step 3: Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# Step 4: Feature scaling
# ------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# Step 5: Create and train Logistic Regression model
# ------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------------------
# Step 6: Make predictions
# ------------------------------
y_pred = model.predict(X_test)

# ------------------------------
# Step 7: Evaluate the model
# ------------------------------
cm = confusion_matrix(y_test, y_pred) # CM calculation
print("Confusion Matrix:\n", cm)
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------------------------------------
# ADDED PART: Confusion Matrix-ah Box (Visual) ah kaata
# ------------------------------------------------------------
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Not Placed', 'Placed'], 
            yticklabels=['Not Placed', 'Placed'])
plt.title('Confusion Matrix Visualization Box')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ------------------------------

# Step 8: Predict placement for a new student

# ------------------------------
new_student = pd.DataFrame([[6, 68, 1]],
                           columns=['Hours_Studied', 'Previous_Score', 'Internship'])

new_student_scaled = scaler.transform(new_student)

placement_pred = model.predict(new_student_scaled)
placement_prob = model.predict_proba(new_student_scaled)

print(f"\nPredicted Placement Status: {'Placed' if placement_pred[0]==1 else 'Not Placed'}")
print(f"Probability of Placement: {placement_prob[0][1]:.2f}")  
*/
```

## Output:

<img width="644" height="338" alt="ML EXP 5 -1" src="https://github.com/user-attachments/assets/fd38b77b-fc9d-4dd0-bee9-b95b34c2e860" />

<img width="716" height="590" alt="ML EXP 5 - 2" src="https://github.com/user-attachments/assets/6d700b1d-ab6c-4a1b-87d0-c95829568a6e" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
