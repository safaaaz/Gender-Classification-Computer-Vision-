import sys
from skimage import feature
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import cv2
import os


categories = ['male', 'female']
# input directories
dir_train= sys.argv[1]
dir_valid= sys.argv[2]
dir_test= sys.argv[3]


class LocalBinaryPatterns:

    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, self.numPoints + 3),range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist


# loading pictures with labels from directory
def loading(dir , desc):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(dir, category)
        for img in os.listdir(path):
            imgpath = os.path.join(path, img)
            image = cv2.imread(imgpath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = desc.describe(gray)
            labels.append(categories.index(category))
            data.append(hist)
    return data, labels


# find max accuracy and the model with this accuracy
def find_max(datatrain, labelstrain, datavalid, labelsvalid, max_accuracy, goodmodel):
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
    svc = svm.SVC(probability=True)
    model = GridSearchCV(svc, param_grid)
    model.fit(datatrain, labelstrain)
    if max_accuracy < model.score(datavalid, labelsvalid):
        max_accuracy = model.score(datavalid, labelsvalid)
        goodmodel=model
    model = svm.SVC(kernel='linear', C=1)
    model.fit(datatrain, labelstrain)
    if max_accuracy < model.score(datavalid, labelsvalid):
        max_accuracy = model.score(datavalid, labelsvalid)
        goodmodel=model
    return max_accuracy, goodmodel


model = svm.SVC(kernel='linear', C=1)
desc = LocalBinaryPatterns(8, 1)
arr_train=loading(dir_train, desc)
arr_valid = loading(dir_valid, desc)
max_accuracy8, model= find_max(arr_train[0], arr_train[1], arr_valid[0], arr_valid[1], 0, model)

desc2 = LocalBinaryPatterns(24, 3)
arr_train=loading(dir_train, desc2)
arr_valid = loading(dir_valid, desc2)
max_accuracy24, model= find_max(arr_train[0], arr_train[1], arr_valid[0], arr_valid[1], max_accuracy8, model)
if max_accuracy24>max_accuracy8:
    desc=desc2

# actual and predicted values for confusion matrix
actual = []
predicted = []

# loop over the testing images
for category in categories:
    path = os.path.join(dir_test, category)
    for img in os.listdir(path):
        image = cv2.imread(os.path.join(path, img))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        prediction = model.predict(hist.reshape(1, -1))
        actual.append(category)
        predicted.append(categories[prediction[0]])

matrix = confusion_matrix(actual, predicted, labels=['male', 'female'])

f = open("results.txt", "w")
f.write("svm with kernel: {}".format(model.kernel))
f.write("\nnumber points: {}".format(desc.numPoints))
f.write("\nradius: {}".format(desc.radius))
f.write("\nAccuracy: {:.2f}%".format(max_accuracy24*100))
f.write("\n\nconfusion matrix: ")
f.write(f"\n \t male \t female \n male \t {matrix[0][0]} \t {matrix[1][0]} \n female \t {matrix[0][1]} \t {matrix[1][1]}")
f.close()
