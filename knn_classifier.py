from ucimlrepo import fetch_ucirepo 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
seed = 47259885
  
X = wine_quality.data.features.to_numpy()
y = wine_quality.data.targets.to_numpy()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, shuffle=True)
print(X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)

ks = np.arange(1, 21, 2)
weights = ['uniform', 'distance']
figure, axes = plt.subplots(1,2, figsize=(8, 8))
training_scores = {'uniform':[],'distance':[]}
testing_scores = {'uniform':[], 'distance':[]}
for k in ks:
    for weight in weights:
        knn = KNeighborsClassifier(k, weights=weight, p=1)
        knn.fit(X_tr, y_tr.ravel())
        train_pred = np.round(knn.predict(X_tr))
        train_score = accuracy_score(y_tr, train_pred)

        test_pred = np.round(knn.predict(X_te))
        test_score = accuracy_score(y_te, test_pred)

        print(f"k={k}, weight={weight}")
        print(f"train score={train_score}, test score={test_score}")
        training_scores[weight].append(train_score)
        testing_scores[weight].append(test_score)

        
axes[0].plot(ks, training_scores['uniform'], color='red')
axes[0].plot(ks, testing_scores['uniform'], color='green')
axes[1].plot(ks, training_scores['distance'], color='red')
axes[1].plot(ks, testing_scores['distance'], color='green')
axes[0].set_title('uniform')
axes[1].set_title('distance')
axes[0].set_xlabel('k values')
axes[0].set_ylabel('accuracy score')
axes[1].set_xlabel('k values')
axes[1].set_ylabel('accuracy score')
plt.show()


# todo: 
# try dropping features (noisy?)

# evaluate using confusion matrix