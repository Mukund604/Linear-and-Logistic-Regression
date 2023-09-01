#Importing the required libraries.
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

#Using the Iris datset from the inbuild datasets from scikit-learn library
iris = datasets.load_breast_cancer()
X = iris.data
y = iris.target

#Splitting the dataset into Training and Testing sets
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#Initializing our model and fitting the given data into the model
clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


#Checking the Accuracy of the model 
def Accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
print(Accuracy(y_test,y_pred))