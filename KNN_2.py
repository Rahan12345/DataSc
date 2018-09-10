# Breast Cancer Classifier

# Importing Breast Cancer Wisconsin (Diagnostic) Data Set from sci-kit learn
from sklearn.datasets import load_breast_cancer

# Loading data
cancer_data = load_breast_cancer()
""" print(type(cancer_data)) """ # If needed

# To divide data into training and validation slots
from sklearn.model_selection import train_test_split

# Using 80% of available data for training, rest for validation
training_data, validation_data, training_labels, validation_labels = \
train_test_split(cancer_data.data, cancer_data.target, train_size = 0.8, random_state = 30)

# Importing KNN Classifer funtion
from sklearn.neighbors import KNeighborsClassifier

# For storing accuracy values obtained from differing values of k
accuracy = []
for k in range(1, len(validation_labels)+1):
    classifier = KNeighborsClassifier(n_neighbors = k)
    # "Fitting" the training data with the labels
    classifier.fit(training_data, training_labels)
    accuracy.append(classifier.score(validation_data, validation_labels))

# For plotting curve between k values and corresponding accuracy
import matplotlib.pyplot as plt

plt.plot([x for x in range(k)], accuracy)
plt.xlabel('k')
plt.ylabel('Classifier Accuracy')
plt.xlim(0,114)
#plt.ylim(0,1)
plt.title('Breast Cancer Classifier Accuracy (with KNN)')
plt.show()
