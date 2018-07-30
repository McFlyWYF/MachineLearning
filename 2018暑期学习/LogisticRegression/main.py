import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%matplotlib inline
plt.rcParams['figure.figsize'] = (15., 12.) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
#%load_ext autoreload
#%autoreload 2


# Read the data for you
data = pd.read_csv('./DSVC/datasets/MNIST.csv',header=0).values # change the path by yourself
imgs = data[0::,1::]
labels = data[::,0]


# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = range(10)
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(labels == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(imgs[idx].reshape(28,28).astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# transform the labels to binary
for i in xrange(len(labels)):
    if labels[i] != 0:
        labels[i] = 1

# 2/3 training set
# 1/3 test set
split_index = len(labels) * 2 / 3
X_train = imgs[:split_index]
y_train = labels[:split_index]
X_test = imgs[split_index:]
y_test = labels[split_index:]

X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

print X_train.shape
print X_test.shape



X_train_feats = None # choose and extract features
X_test_feats = None # choose and extract features
X_train_feats = (X_train - np.mean(X_train)) / np.std(X_train)
X_test_feats = (X_test - np.mean(X_test)) / np.std(X_test)

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = range(2)
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(labels == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(imgs[idx].reshape(28,28).astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


from DSVC.classifiers import LogisticRegression

# Start training.
# You should open DSVC/classifiers/logistic_regression.py and implement the function.
# Then run this cell.

classifier = LogisticRegression()
loss_history = classifier.train(
    X_train,
    y_train,
    learning_rate = 1e-3,
    num_iters = 500,
    batch_size = 64,
)


plt.plot(loss_history)


y_test_pred = classifier.predict(X_test)
print "The accuracy socre is ", np.mean(y_test == y_test_pred)

TP = 0.0
TN = 0.0
FP = 0.0
FN = 0.0

# compute TP
for i in range(785):
    if y_test_pred[i] == 0 and y_test[i] == 0:
        TP += 1.0

# compute TN
for i in range(785):
    if y_test_pred[i] == 1 and y_test[i] == 1:
        TP += 1.0

# compute FP
for i in range(785):
    if y_test_pred[i] == 0 and y_test[i] == 1:
        TP += 1.0

# compute FN
for i in range(785):
    if y_test_pred[i] == 1 and y_test[i] == 0:
        TP += 1.0

precision = TP / (TP + FP)
recall = TP / (TP + FN)
print precision
print recall
print 'F1:', precision * recall * 2 / (precision + recall)
# print precision
# print recall
# print 'F1:', precision*recall*2/(precision+recall)

# Read the data for you
data = pd.read_csv('./DSVC/datasets/MNIST.csv', header=0).values  # change the path by yourself
imgs = data[0::, 1::]
labels = data[::, 0]

# 2/3 training set
# 1/3 test set
split_index = len(labels) * 2 / 3
X_train = imgs[:split_index]
y_train = labels[:split_index]
X_test = imgs[split_index:]
y_test = labels[split_index:]

X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

print X_train.shape
print X_test.shape



X_train_feats = None # choose and extract features
X_test_feats = None # choose and extract features
X_train_feats = (X_train - np.mean(X_train)) / np.std(X_train)
X_test_feats = (X_test - np.mean(X_test)) / np.std(X_test)



# Start training.
# You should update your code in DSVC/classifiers/logistic_regression.py .
# Then run this cell.

classifier = LogisticRegression()
classifier.one_vs_all(
    X_train_feats,
    y_train,
    learning_rate = 1e-3,
    num_iters = 500,
    batch_size = 64,
)


# you may change your code in function `predict`
y_test_pred = classifier.predict(X_test_feats)
print "The accruacy socre is ", np.mean(y_test == y_test_pred)
