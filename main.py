import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1000 student random dataset
np.random.seed(42)
num_students = 10000

data = {
    'CGPA': np.random.uniform(5.0, 10.0, num_students),
    'Study_Hours': np.random.uniform(1, 10, num_students),
    'Attendance': np.random.uniform(60, 100, num_students),
    'Projects_Done': np.random.randint(0, 5, num_students),
}

df = pd.DataFrame(data)
print(df.info())

# Logic for Target: Placement (1 = Yes, 0 = No)
# A student is placed if they have high CGPA, good attendance, and projects.
# We add some 'Noise' to make it realistic.
noise = np.random.normal(0, 1, num_students)
score = (df['CGPA'] * 2) + (df['Attendance'] * 0.1) + (df['Projects_Done'] * 3) + noise
df['Placed'] = (score > 25).astype(int)

print("--- Dataset Sample ---")
print(df.head())
print("\n")

# PREPROCESSING
# Selecting Features (X) and Target (y)
X = df[['CGPA', 'Study_Hours', 'Attendance', 'Projects_Done']]
y = df['Placed']

# ==========================================
# STEP 3: TRAIN-TEST SPLIT
# ==========================================
# 80% for training, 20% for testing the model's 'unseen' performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL TRAINING
# Using Logistic Regression (standard for Binary Classification)
model = LogisticRegression()
model.fit(X_train, y_train)

# PREDICTION & EVALUATION
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"--- Model Results ---")
print(f"Accuracy Score: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# --- Visualizing Results ---
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 1. Confusion Matrix Heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

# 2. Feature Importance
importance = model.coef_[0]
features = X.columns
sns.barplot(x=importance, y=features, ax=ax[1], palette='viridis')
ax[1].set_title('Feature Importance (Model Coefficients)')

plt.tight_layout()
plt.show()

# CUSTOM PREDICTION (INFERENCE)
def predict_my_placement(cgpa, hours, attendance, projects):
    data = [[cgpa, hours, attendance, projects]]
    new_student_df = pd.DataFrame(data, columns=['CGPA', 'Study_Hours', 'Attendance', 'Projects_Done'])
    prediction = model.predict(new_student_df)
    probability = model.predict_proba(new_student_df)[0][1]
    result = "PLACED" if prediction[0] == 1 else "NOT PLACED"
    print(f"\n--- Custom Prediction ---")
    print(f"Input: CGPA={cgpa}, Hours={hours}, Attendance={attendance}%, Projects={projects}")
    print(f"Result: {result} (Probability: {probability*100:.2f}%)")

# Test with your own data!
predict_my_placement(cgpa=8.5, hours=6, attendance=90, projects=2)
predict_my_placement(cgpa=6.0, hours=2, attendance=65, projects=0)