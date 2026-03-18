from ucimlrepo import fetch_ucirepo 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
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
y = y_raw.to_numpy()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, shuffle=True)
print(X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)


# k and weight value investigation
# ks = [1, 10, 50, 100, 250, 500]
# weights = ['uniform', 'distance']
# figure, axes = plt.subplots(1,2, figsize=(8, 8))
# training_scores = {'uniform':[],'distance':[]}
# testing_scores = {'uniform':[], 'distance':[]}
# for k in ks:
#     for weight in weights:
#         knn = KNeighborsClassifier(k, weights=weight, p=2)
#         knn.fit(X_tr, y_tr.ravel())
#         train_pred = np.round(knn.predict(X_tr))
#         train_score = accuracy_score(y_tr, train_pred)

#         test_pred = np.round(knn.predict(X_te))
#         test_score = accuracy_score(y_te, test_pred)

#         print(f"k={k}, weight={weight}")
#         print(f"train score={train_score}, test score={test_score}")
#         training_scores[weight].append(train_score)
#         testing_scores[weight].append(test_score)

        
# axes[0].plot(ks, training_scores['uniform'], color='red')
# axes[0].plot(ks, testing_scores['uniform'], color='green')
# axes[1].plot(ks, training_scores['distance'], color='red')
# axes[1].plot(ks, testing_scores['distance'], color='green')
# axes[0].set_title('uniform')
# axes[1].set_title('distance')
# axes[0].set_xlabel('k values')
# axes[0].set_ylabel('accuracy score')
# axes[1].set_xlabel('k values')
# axes[1].set_ylabel('accuracy score')
# plt.show()


# feature dropping
# ks = [100, 200, 300]
# figure, axes = plt.subplots(1, figsize=(6, 6))
# plt.rcParams['font.size'] = 10
# testing_scores = {}
# col_test_scores = []

# for i, col in enumerate(X_raw.columns):
#     testing_scores[col] = []
#     for k in ks:
#         # choose p=1 because focusing on dropping noisy/irrelevant features
#         knn = KNeighborsClassifier(k, weights='distance', p=1)
#         cur_X_tr = np.delete(X_tr, i, axis=1)
#         cur_X_te = np.delete(X_te, i, axis=1)
#         knn.fit(cur_X_tr, y_tr.ravel())
#         test_pred = np.round(knn.predict(cur_X_te))
#         test_score = accuracy_score(y_te, test_pred)

#         print(f"k={k}, dropped {col}")
#         print(f"test score={test_score}")
#         testing_scores[col].append(test_score)

#     col_test_scores.append(np.max(testing_scores[col]))

# axes.plot(X_raw.columns, col_test_scores, color='red')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()
# dropping sulfates gives best results


# testing k value for dataset without sulphates
ks = np.arange(1, 501, 25)
figure, axes = plt.subplots(1, figsize=(8, 8))
testing_scores = []
for k in ks:
    # choose p=1 because focusing on dropping noisy/irrelevant features
    knn = KNeighborsClassifier(k, weights='distance', p=1)
    cur_X_tr = np.delete(X_tr, [9], axis=1)
    cur_X_te = np.delete(X_te, [9], axis=1)
    knn.fit(cur_X_tr, y_tr.ravel())

    test_pred = np.round(knn.predict(cur_X_te))
    test_score = accuracy_score(y_te, test_pred)

    print(f"k={k}")
    print(f"test score={test_score}")
    testing_scores.append(test_score)
axes.plot(ks, testing_scores, color='red')
axes.set_xlabel('k values')
axes.set_ylabel('accuracy score')
plt.show()

# test with bootstrapping
# evaluate using confusion matrix