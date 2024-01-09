#%%
import matplotlib.pyplot as plt 
import numpy as np
from skimage.feature import hog 
from sklearn import datasets 
from mlxtend.data import loadlocal_mnist 
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score



lb = LabelBinarizer()

##load database mnist
train_images, train_labels = loadlocal_mnist(images_path='mnist_dataset/train-images.idx3-ubyte', 
                                             labels_path='mnist_dataset/train-labels.idx1-ubyte')

test_images, test_labels = loadlocal_mnist(images_path='mnist_dataset/t10k-images.idx3-ubyte', 
                                             labels_path='mnist_dataset/t10k-labels.idx1-ubyte')


# print(train_images[0].shape)

# plt.imshow(train_images[0].reshape(28,28), cmap='gray')

# print(train_labels[0])

##test ekstrak hog feature
feature, hog_img = hog(train_images[0].reshape(28,28), orientations=9, pixels_per_cell= (8,8), cells_per_block= (2,2), visualize=True, block_norm='L2' )

# print(feature)
# print(feature.shape)

##lihat histogram
# plt.bar(list(range(feature.shape[0])), feature)

##preposisi menggunankan hog feature
##buat variale untuk julah dimensi dan jumlah sample
n_dims = feature.shape[0]
n_samples = train_images.shape[0]


x_train, y_train = datasets.make_classification(n_samples=n_samples, n_features=n_dims)
# print(x_train.shape)


for i in range(n_samples):
    x_train[i], _ = hog(train_images[i].reshape(28,28), orientations=9, pixels_per_cell= (8,8), cells_per_block= (2,2), visualize=True, block_norm='L1')
    y_train[i]= train_labels[i]

lb.fit(y_train)
y_train_one_hot = lb.transform(y_train)

# print(y_train_one_hot[0])
# print(y_train[0])

# label = lb.inverse_transform(np.array([y_train_one_hot[0]]))

# print(label[0])

###################################################################

clf = MLPClassifier(hidden_layer_sizes=(128,64,10),solver='sgd', learning_rate_init= 0.001, max_iter=200)
# print(clf.fit(x_train, y_train_one_hot))
clf.fit(x_train, y_train_one_hot)

n_samples = test_images.shape[0]
x_test, y_test = datasets.make_classification(n_samples=n_samples, n_features=n_dims)

for i in range(n_samples):
    x_test[i], _ = hog(test_images[i].reshape(28,28), orientations=9, pixels_per_cell= (8,8), cells_per_block= (2,2), visualize=True, block_norm='L2')
    y_test[i] = test_labels[i]


y_test_one_hot = lb.transform(y_test)

clf.fit(x_test, y_test)
y_pred_one_hot = clf.predict(x_test)

# y_pred = lb.inverse_transform(y_pred_one_hot[0])

plt.imshow(test_images[20].reshape(28,28), cmap='gray')
out_one_hot = clf.predict(x_test[20].reshape(10, n_dims))
out = lb.inverse_transform(out_one_hot.reshape(10,10))

print(out)

##confusion matrix
conf_mat = confusion_matrix(y_test,lb.inverse_transform(y_pred_one_hot[0]))
class_name = ['0','1','2','3','4','5','6','7','8','9']
fig, x = plot_confusion_matrix(conf_mat=conf_mat)

##accuracy
acc = accuracy_score(y_test,lb.inverse_transform(y_pred_one_hot[0]))

##pression
pression = precision_score(y_test,lb.inverse_transform(y_pred_one_hot[0]), average=None)
print(pression)


# %%
