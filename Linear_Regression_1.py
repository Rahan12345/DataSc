""" Linear Regression """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

a = np.array([x for x in range(0,100,2)])
b = []
for i in range(len(a)):
    b.append(np.random.normal(a[i]+np.random.randint(1,4,1),6,1))

a = a.reshape(-1,1)

line_fitting = LinearRegression()

line_fitting.fit(a,b)

y_predicted = line_fitting.predict(a)

plt.plot(a,b,'o')
plt.plot(a,y_predicted)

plt.show()