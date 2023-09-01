#Importing the libraires
import numpy as np

class LogisticRegression:
    #Making a constructor for the Logistic Regression class
    def __init__(self, lr = 0.0001, n = 1000):
        self.lr = lr
        self.n = n
        self.m = None
        self.c = None
        
    #Fitting the Model with the given data
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.m = np.zeros(n_features)
        self.c = 0
        
        #Starting our Gradient Descent 
        for _ in range(self.n):
            linearModel = np.dot(X,self.m) + self.c
            y_pred = self._sigmoid(linearModel)
            
            #Finding the local Minima by taking derivatives of the weights and bais
            dm = (1/n_samples) * np.dot(X.T, (y_pred - y))
            dc = (1/n_samples) * np.sum(y_pred - y)
        
            #Updating the weights and bais for making the optimized model
            self.m -= self.lr * dm
            self.c -= self.lr * dc
            
    #Making predicitions from the trained model that we created
    def predict(self, X):
        linearModel = np.dot(X, self.m) + self.c
        y_pred = self._sigmoid(linearModel)
        clf = [1 if i > 0.5 else 0 for i in y_pred]
        return clf
    
    #Sigmoid function for making continous output from our model and map it to a value between 0 and 1.
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))