import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def computeCost(x, y, theta):
    
    m = len(y)
    prediction = np.matmul(x,theta)
    sqrError = np.square(np.subtract(prediction, y))
    J = ( np.sum(sqrError, dtype=float) ) / ( 2*m )
    return(J)

def gradientDescent(x, y, theta, alpha, iterations):

    m = len(y)

    for i in range(0,iterations):

        theta = theta - ( alpha * ( x.T.dot(np.matmul(x,theta) -  y) ) ) / m  
        
    return(theta)

#plotting the data file in graph
print("Graph of data is now going to be plotted")

x, y = np.loadtxt("ex1data1.txt", dtype={ 'names': ('x','y') , 'formats': (np.float, np.float) }, delimiter = ',' , skiprows=0, unpack=True, ndmin=2 )
no_of_iterations = len(y) 

plt.scatter(x, y , s= 20, c='r', marker='x')
plt.xlabel('population size in 10,000s')
plt.ylabel('profit in $10,000s')
plt.show()

#adding coulumn of 1's in x so that x[:,0] == 1, satisfying the equation h = theta[0] * x[:,0] + theta[1] * x[:,1]
z = np.ones((no_of_iterations,1), dtype=float)
x = np.insert(x, 0, 1, axis=1)

#initializing parameters of cost function and gradient descent
theta = np.zeros( (2,1) , dtype=float)
alpha = 0.01
iterations = 1500

#calculating cost function
print('Testing the cost function')
J = computeCost(x, y, theta)
print(J)
print("Expected Cost is 32.07")

J = computeCost(x, y, [[-1], [2]])
print(J)
print("Expected Cost is 54.24")

#applying gradient descent
print("Now its time for gradient descent")
theta = gradientDescent(x, y, theta, alpha, iterations)

#plotting data set and gradient descent predictions
plt.scatter(x[:,1], y , s= 20, c='r', marker='x')
plt.xlabel('population size in 10,000s')
plt.ylabel('profit in $10,000s')
plt.plot(x[:,1], np.matmul(x,theta), label='Linear regression (Gradient descent)')
plt.show()