#  tools we will be needing for the project
#  using scikit-learn bc it would be perfect for our size of data.
from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

wine_quality = fetch_ucirepo(id=186) # here we are fetching wine quality dataset firectely form UCI repository using 186 as ID

seed = 47259885   # this helps getting the same results every time we run it. It is imp that our numbers dont get chnaged for our report.

X_raw = wine_quality.data.features  # here we are pulling out chemical deatrures and the quality scores.
y_raw = wine_quality.data.targets

scaler = StandardScaler()   # we are using standardscaler to make sure that every feature has a mean of 0 and a var of 1
X = scaler.fit_transform(X_raw.to_numpy())
y = y_raw.to_numpy().ravel() 

# we split our data into three piles: traning 60% validation 20% and testing 20%
X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.40, random_state=seed, shuffle=True)
X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, shuffle=True)

# here we choose 1 hidden layer with 100 neurons
#  as it gave the best balance between learning and speed.
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1500, random_state=seed)  # here max_iter = 1500 to give model enough time to find best patterns

mlp.fit(X_tr, y_tr) # learing happens here

test_preds = mlp.predict(X_te)  # giving model tghe test set

print(f"Final Test Accuracy: {accuracy_score(y_te, test_preds):.4f}")  # printing final scores.

# this report shows us where the model struggled, like with the very rare high-quality or low-quality wines.
print("\nClassification Report:")
print(classification_report(y_te, test_preds, zero_division=0))