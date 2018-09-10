# Breast Cancer Classifier

from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

from sklearn.model_selection import train_test_split

training_data, validation_data, training_labels, validation_labels = \
train_test_split(cancer_data.data, cancer_data.target, train_size = 0.8, random_state = 30)

from sklearn.neighbors import KNeighborsClassifier

accuracy = []
for k in range(1, len(validation_labels)+1):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(training_data, training_labels)
    accuracy.append(classifier.score(validation_data, validation_labels))

import matplotlib.pyplot as plt

plt.plot([x for x in range(k)], accuracy)
plt.xlabel('k')
plt.ylabel('Classifier Accuracy')
plt.xlim(0,114)
#plt.ylim(0,1)
plt.title('Breast Cancer Classifier Accuracy (with KNN)')
plt.show()