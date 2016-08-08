#Google Devlopers demo from youtube
#https://www.youtube.com/watch?v=tNa99PG8hR8
#No visualization plugin (graphviz)

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()


test_idx = [0,50,100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target



