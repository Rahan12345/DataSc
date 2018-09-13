# K-Means Clustering of Iris dataset using sklearn
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Loading iris dataset
iris = datasets.load_iris()

# Utilizing all data in iris dataset
samples = iris.data

model = KMeans(n_clusters=3)

# Training model
model.fit(samples)

# Labelling
labels = model.predict(samples)

"""print(labels)"""

# Make a scatter plot of x and y and using labels to definte the colors
x = samples[:,0]
y = samples[:,1]

# Plotting the different labels
plt.scatter(x,y,c=labels,alpha=0.5)

plt.show()
