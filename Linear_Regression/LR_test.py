#Importing the Libraries
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearReg import LinearRegression
from sklearn.model_selection import train_test_split
 
#Making a Synthetic Dataset for Training the model
X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=30, random_state=123)

#Splitting the model into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#Training the model and making predicitions
regressor = LinearRegression(lr = 0.001, n = 10000)
regressor.fit(X_train, y_train)
predicitons = regressor.predict(X_test)


#Calculating the accuracy of the model using Mean Squared Error Method
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
print(mse(y_test, predicitons))

#Plotting the graph 
y_pred_line = regressor.predict(X)
plt.scatter(X_train, y_train, color='y', s = 20)
plt.scatter(X_test, y_test, color='black', s = 20)
plt.plot(X, y_pred_line, linewidth = 2, color='red')
plt.show()