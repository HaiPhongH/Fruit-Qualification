import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from lbp import getLBPCoefficents
from svm import MulticlassSVM

import cv2, os
from imutils import paths
import joblib
from preprocess import preprocess

# INITILAZE DATA AND LABELS LIST FPR TRAINING

data_train = []
labels_train = []

# TRAINING PHASE
for imagePath in paths.list_images('../dataset/training/'):
    # Get the image of poison part 
    image = preprocess(imagePath)

    # Extract features for the image of poison part
    features = getLBPCoefficents(image)

    # Append features and labels of training-set to arrays
    labels_train.append(imagePath.split(os.path.sep)[-2])
    data_train.append(features)

# TRAINING MODEL WITH SKLEARN LIBRARY
model = SVC(C=100, random_state=42)
model.fit(data_train, labels_train)

# ACCURANCY
print("Accuracy: %.2f %%" %(100*model.score(data_train, labels_train)))

# TRAINING MODEL WITH MULTIPLE SVM
arr_data = np.asarray(data_train, dtype=None)
arr_labels = np.asarray(labels_train, dtype=None)

clf = MulticlassSVM(C=0.1, tol=0.01, max_iter=100, random_state=0, verbose=1)
clf.fit(arr_data, arr_labels)

# ACCURANCY
print("Accuracy: %.2f %%" %(100*clf.score(arr_data, arr_labels)))

# SAVE MODEL TO A FILE
filename = 'final_model.sav'
joblib.dump(model, filename)

# INITILAZE DATA AND LABELS LIST FPR TRAINING
data_test = []
labels_test = []

# TESTING PHASE
for imagePath in paths.list_images('../dataset/testing/'):

    image = preprocess(imagePath)
    features = getLBPCoefficents(image)

    labels_test.append(imagePath.split(os.path.sep)[-2])
    data_test.append(features)

    # SHOW PREDICTION ON IMAGE

    # prediction = model.predict(features.reshape(1, -1))

    # cv2.putText(image_ori, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
	# 	1.0, (0, 0, 255), 3)
    # cv2.imshow("Image", image_ori)
    # cv2.waitKey(0)

# ACCURANCY OF TESTING PHASE
labels_pred = model.predict(data_test)
print("Accuracy: %.2f %%" %(100*accuracy_score(labels_test, labels_pred)))



