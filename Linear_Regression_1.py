""" Linear Regression """

# Importing modules for reusability (Will be uploading the code without using sklearn)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Creating the x and y coordinates for data points
a = np.array([x for x in range(0,100,2)])
b = []
for i in range(len(a)):
    b.append(np.random.normal(a[i]+np.random.randint(1,4,1),6,1))

# Reshaping the x coordinate as a 1xN array from Nx1 array    
a = a.reshape(-1,1)

# Accessing the functions available to LinearRegression
line_fitting = LinearRegression()

# Finding the best values of m and b for best fit or least error
line_fitting.fit(a,b)

# Predicting the corresponding y coordinate values using the values of slope and intercept
y_predicted = line_fitting.predict(a)

# Plotting the data
plt.plot(a,b,'o')
plt.plot(a,y_predicted)

# Displaying the plot
plt.show()
