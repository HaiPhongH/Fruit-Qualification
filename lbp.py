import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,plot_confusion_matrix

from sklearn.metrics import accuracy_score
from lbp import getLBPCoefficents

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

# Calculate and plot confusion matrix
disp = plot_confusion_matrix(model, data_test, labels_test,
                                cmap=plt.cm.Blues,
                                normalize='true')

print(disp.confusion_matrix)
plt.show()
