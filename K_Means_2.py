import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()

samples = iris.data

model = KMeans(n_clusters=3)

model.fit(samples)

labels = model.predict(samples)

"""print(labels)"""

# Make a scatter plot of x and y and using labels to definte the colors
x = samples[:,0]
y = samples[:,1]

plt.scatter(x,y,c=labels,alpha=0.5)

plt.show()
