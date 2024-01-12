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
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



lb = LabelBinarizer()

##load database mnist
train_images, train_labels = loadlocal_mnist(images_path='mnist_dataset/train-images.idx3-ubyte', 
                                             labels_path='mnist_dataset/train-labels.idx1-ubyte')

test_images, test_labels = loadlocal_mnist(images_path='mnist_dataset/t10k-images.idx3-ubyte', 
                                            labels_path='mnist_dataset/t10k-labels.idx1-ubyte')

# print(train_images.shape)
# plt.imshow(train_images[0].reshape(28,28),cmap='gray')


# # ##extract_hog_features
feature, hog_img = hog(train_images[0].reshape(28,28), orientations=9, pixels_per_cell= (8,8), cells_per_block= (2,2), visualize=True, block_norm='L2' )
# print(hog_img.shape)
# plt.bar(list(range(feature.shape[0])), feature)


# # ##preposisi menggunakan hog feature
# # ##buat variale untuk julah dimensi dan jumlah sample
n_dims = feature.shape[0]
n_samples = train_images.shape[0]
x_train, y_train = datasets.make_classification(n_samples=n_samples, n_features=n_dims)
# # # print(x_train.shape)


# # ##generate dataset berupa hog feature
for i in range(n_samples):
    x_train[i], _ = hog(train_images[i].reshape(28,28), orientations=9, pixels_per_cell= (8,8), cells_per_block= (2,2), visualize=True, block_norm='L1')
    y_train[i]= train_labels[i]


# # ##convert categorical label to one hot label
lb.fit(y_train)
y_train_one_hot = lb.transform(y_train)
# print(y_train_one_hot.shape)
# print(y_train_one_hot[0])
# print(y_train[0])
label = lb.inverse_transform(np.array([y_train_one_hot[0]]))
# # print(label[0])

# ##klasifikasi neural network
clf = MLPClassifier(hidden_layer_sizes=(128,64,10),solver='sgd', momentum=0.9, beta_1=0.9, learning_rate_init= 0.001, max_iter=100)
# ##melakukan proses trainning
clf.fit(x_train, y_train_one_hot)

# ##prediksi dataset test
n_samples = test_images.shape[0]
x_test, y_test = datasets.make_classification(n_samples=n_samples, n_features=n_dims)

for i in range(n_samples):
    x_test[i], _ = hog(test_images[i].reshape(28,28), orientations=9, pixels_per_cell= (8,8), cells_per_block= (2,2), visualize=True, block_norm='L2')
    y_test[i] = test_labels[i]

y_test_one_hot = lb.transform(y_test)


plt.imshow(test_images[1].reshape(28,28), cmap='gray')
out_one_hot = clf.predict(x_test[1].reshape(1,n_dims))

out = lb.inverse_transform(out_one_hot.reshape(1,10))
print(out)


# # Standardize features
scaler = StandardScaler().fit(x_train)
X_train_hog = scaler.transform(x_train)
X_test_hog = scaler.transform(x_test)

# # Train SVM
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_hog, y_train)
# Predict
y_pred = svm_model.predict(X_test_hog)


# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

# Display results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Precision:", precision)

# %%
