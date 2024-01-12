import cv2
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from skimage.feature import hog
from skimage import exposure

digits = datasets.load_digits()

hog_features = []
for image in digits.images:
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualize=True)
    hog_features.append(fd)

hog_features = np.array(hog_features)
labels = digits.target

X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.5, random_state=40)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

#menampilkan hasil
print("Confusion Matrix:")
print(conf_matrix)
print("\nAkurasi:", accuracy)
print("\nPresisi:", precision)
