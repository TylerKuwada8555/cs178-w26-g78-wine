from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fetch dataset
wine_quality = fetch_ucirepo(id=186)
seed = 47259885

X_raw = wine_quality.data.features
y_raw = wine_quality.data.targets

X = scaler.fit_transform(X_raw.to_numpy())
y = y_raw.to_numpy().ravel()

# 70/15/15 split: train / val / test
X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.40, random_state=seed, shuffle=True)
X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, shuffle=True)

print(f"Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}")

# Vary n_estimators and max_depth
n_estimators_list = [10, 50, 100, 200, 300]
max_depths = [None, 5, 10, 20]

best_val_score = 0
best_params = {}
results = []

for n in n_estimators_list:
    for d in max_depths:
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=seed)
        rf.fit(X_tr, y_tr)

        val_score = accuracy_score(y_val, rf.predict(X_val))
        results.append((n, d, val_score))
        print(f"n_estimators={n}, max_depth={d} -> val accuracy={val_score:.4f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_params = {'n_estimators': n, 'max_depth': d}

print(f"\nBest params: {best_params}, val accuracy: {best_val_score:.4f}")


# Final evaluation on test set using best params
best_rf = RandomForestClassifier(**best_params, random_state=seed)
best_rf.fit(X_tr, y_tr)

train_score = accuracy_score(y_tr, best_rf.predict(X_tr))
test_score = accuracy_score(y_te, best_rf.predict(X_te))
print(f"\nFinal train accuracy: {train_score:.4f}")
print(f"Final test accuracy:  {test_score:.4f}")


# Confusion matrix
cm = confusion_matrix(y_te, best_rf.predict(X_te))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Random Forest Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("output/rf_confusion_matrix.png")
plt.show()


# Feature importance
importances = best_rf.feature_importances_
plt.figure(figsize=(10, 5))
plt.bar(X_raw.columns, importances)
plt.xticks(rotation=90)
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("output/rf_feature_importance.png")
plt.show()