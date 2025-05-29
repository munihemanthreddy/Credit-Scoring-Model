!pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
data_size = 1000
data = pd.DataFrame({
    'income': np.random.normal(50000, 15000, data_size),
    'total_debt': np.random.normal(15000, 5000, data_size),
    'credit_score': np.random.normal(650, 50, data_size),
    'late_payments': np.random.poisson(2, data_size),
    'loan_amount': np.random.normal(10000, 3000, data_size),
    'employment_status': np.random.choice(['employed', 'unemployed'], data_size, p=[0.8, 0.2]),
    'credit_history_len': np.random.normal(10, 3, data_size),
    'open_accounts': np.random.poisson(5, data_size)
})

data['target'] = ((data['credit_score'] > 620) &
                  (data['late_payments'] < 3) &
                  (data['income'] > data['total_debt'])).astype(int)

data['employment_status'] = data['employment_status'].map({'employed': 1, 'unemployed': 0})
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    results[name] = (y_test, y_proba)

plt.figure(figsize=(10, 6))
for name, (y_true, y_proba) in results.items():
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_true, y_proba):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves for Credit Scoring Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()
