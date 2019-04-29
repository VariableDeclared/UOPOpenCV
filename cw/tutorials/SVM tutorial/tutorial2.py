import numpy as np
import pandas as pd
from subprocess import check_output
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

digits = datasets.load_digits()
print("Digits dataset keys \n {}".format(digits.keys()))
print("dataset target name: \n {}".format(digits.target_names))
print("shape of dataset: {} \n and target: {}".format(digits.data.shape, digits.target.shape))
print("shape of the images: {}".format(digits.images.shape))

for i in range(0, 4):
    plt.subplot(2, 4, i + 1)
    plt.axis("off")
    imside = int(np.sqrt(digits.data[i].shape[0]))
    im1 = np.reshape(digits.data[i], (imside, imside))
    plt.imshow(im1, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title("Training: {}".format(digits.target[i]))

plt.show()

for i in range(0,4):
    plt.subplot(2, 4,i + 1)
    plt.axis('off')
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: {}'.format(digits.target[i]))


plt.show()

n_samples = len(digits.images)
data_images = digits.images.reshape((n_samples, -1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_images, digits.target)
print("Training data and target sizes: \n {}, {}".format(X_train.shape, y_train.shape))
print("Test data and target sizes: {}\n, {}".format(X_test.shape, y_test.shape))

classifier = svm.SVC(gamma=0.001)
# fit to the training data
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Classification report for the classifier %s \n %s \n" % (classifier, metrics.classification_report(y_test, y_pred)))
print("Confustion matrix: \n %s" % metrics.confusion_matrix(y_test, y_pred))

