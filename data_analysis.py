
# 1️⃣ Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 2️⃣ Create DataFrame
df = pd.DataFrame({
'age': [8, 89, 76, 13, 19, 26, 34, 43, 53, 64, 5, 90, 110, 130],
'Policy': [0, 1, 1, 0, np.nan, 0, 0, 0, 1, 1, np.nan, 1, 1, 1]
})

# 3️⃣ Data Cleaning
df['Policy'].fillna(0, inplace=True)

# 4️⃣ Visualization
plt.scatter(df['age'], df['Policy'], marker='+', color='red')
plt.xlabel("Age")
plt.ylabel("Policy")
plt.title("Model Suitability")
plt.show()

# 5️⃣ Prepare Data for ML
X = df[['age']]
y = df['Policy']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 6️⃣ Logistic Regression Model
model = LogisticRegression()
model.fit(x_train, y_train)

# 7️⃣ Predictions
y_pred = model.predict(x_test)

# 8️⃣ Accuracy and Confusion Matrix
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
#print mera naam 😄🥰😉
print("my name is niteesh Pandey")

print("Test Accuracy:", acc)
print("Confusion Matrix:\n", cm)

# 9️⃣ Optional: Plot Sigmoid Curve
ages = np.linspace(min(df['age']), max(df['age']), 100).reshape(-1,1)
prob = model.predict_proba(ages)[:,1]

plt.scatter(df['age'], df['Policy'], marker='+', color='red', label='Data')
plt.plot(ages, prob, color='blue', label='Sigmoid Curve')
plt.xlabel('Age')
plt.ylabel('Policy Probability')
plt.title('Logistic Regression Fit')
plt.legend()
plt.show()
