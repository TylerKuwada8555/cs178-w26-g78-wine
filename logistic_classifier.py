from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets.values.ravel()

# Split: 60% train, 20% val, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Trying a different C values on validation set
C_values = [0.01, 0.1, 1, 10, 100]
best_C = None
best_val_acc = 0

for C in C_values:
    model = LogisticRegression(max_iter=1000, C=C)
    model.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"C={C}: Validation Accuracy = {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_C = C

# Train final model with best C
print(f"\nBest C: {best_C}")
final_model = LogisticRegression(max_iter=1000, C=best_C)
final_model.fit(X_train, y_train)

test_acc = accuracy_score(y_test, final_model.predict(X_test))
print(f"Final Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, final_model.predict(X_test)))