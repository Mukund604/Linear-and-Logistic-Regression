#Importing the libraries
import numpy as np

class LinearRegression:
    #Making a Constructor for the class
   def __init__(self, lr = 0.001, n = 1000):
      self.lr = lr
      self.n = n
      self.m = None
      self.c = None
      
    #Fit Method
   def fit(self, X, y):
      n_samples,n_featrues = X.shape
      self.m = np.zeros(n_featrues)
      self.c = 0
      
      #Starting our Gradient Descent 
      for _ in range(self.n):
         y_pred = np.dot(X, self.m) + self.c
         
         #Updating our slope weights and bias as we proceed with the gradient descent
         dm = (1/n_samples) * np.dot(X.T, (y_pred - y))
         dc = (1/n_samples) * np.sum(y_pred - y)
         
         
         self.m -= self.lr * dm
         self.c -= self.lr * dc
         
    #Making predicitions from the trained model
   def predict(self, X):
      y_pred = np.dot(X, self.m) + self.c
      return y_pred