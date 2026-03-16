# https://archive.ics.uci.edu/dataset/186/wine+quality

from ucimlrepo import fetch_ucirepo 
import numpy as np
import matplotlib.pyplot as plt

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 

means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
mins = np.min(X, axis=0)
maxes = np.max(X, axis=0)

def basic_data():
    print(f"Datapoints: {X.shape[0]}") 
    print(f"Features: {X.shape[1]}") 
    print()
    print("Feature list: " + ', '.join(X.columns))
    print("Target min/max: " + str(np.min(y)) + '-' + str(np.max(y)))
    print()
    
def feature_data():
    print("Feature means")
    print(means)
    print("Feature STD")
    print(stds)
    print("Feature mins")
    print(mins)
    print("Feature maxes")
    print(maxes)

def graphs():
    figure, axes = plt.subplots(3, 4, figsize=(8, 8))

    for i, col in enumerate(X.columns):
        axes[i // 4][i % 4].scatter(X[col], y)
        axes[i // 4][i % 4].set_xlabel(col)
        axes[i // 4][i % 4].set_ylabel('Score')
    plt.tight_layout()
    plt.show()

print("DATA ANALYSIS".center(25, '-'))

#basic_data()
#feature_data()
graphs()
