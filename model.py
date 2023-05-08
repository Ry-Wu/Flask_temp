import numpy as np
from sklearn import datasets, neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

iris = datasets.load_iris()
cat_names = np.array(['setosa', 'versicolor', 'virginica'])
y = cat_names[iris.target]

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, y, test_size=0.3, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# Make a pickle file of the model
pickle.dump(clf, open('model.pkl', 'wb'))

