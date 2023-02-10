import mnist_reader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os.path
import time
import pickle
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def read_data():
    x_train, y_train = mnist_reader.load_mnist('', kind='train')
    x_test, y_test = mnist_reader.load_mnist('', kind='t10k')
    class_names = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return x_train, y_train, x_test, y_test, class_names


def write_On_file(file_name, data):
    directory = './outputFiles/'
    file_path = os.path.join(directory, file_name + ".txt")
    if not os.path.isdir(directory):
        os.mkdir(directory)
    file = open(file_path, "a+")
    file.write(data)
    file.close()


def store_models(fileName, model):
    directory = './Models/'
    file_path = os.path.join(directory, fileName)
    pickle.dump(model, open(file_path, 'wb'))


def restore_models(fileName):
    directory = './Models/'
    file_path = os.path.join(directory, fileName)
    pickled_model = pickle.load(open(file_path, 'rb'))
    return pickled_model


def scatter_plot_of_training_samples(x_train, y_train):
    y_train = y_train.reshape(60000, 1)
    input_data = x_train / 255
    target = y_train
    # Create a random generator, so to decreases potential biases in the data
    features = ['pixel' + str(i + 1) for i in range(input_data.shape[1])]
    pca_df = pd.DataFrame(input_data, columns=features)
    # Add an additional column 'y', identical with label values in data
    pca_df['label'] = target
    rand = np.random.permutation(pca_df.shape[0])

    results = []
    # Loop through all label
    for i in range(pca_df.shape[0]):
        # Extract the label for comparison
        if pca_df['label'][i] == 0:
            # Save meaningful label to the results
            results.append('T-shirt/top')
        # Following the same code pattern as the one above
        elif pca_df['label'][i] == 1:
            results.append('Trouser')
        elif pca_df['label'][i] == 2:
            results.append('Pullover')
        elif pca_df['label'][i] == 3:
            results.append('Dress')
        elif pca_df['label'][i] == 4:
            results.append('Coat')
        elif pca_df['label'][i] == 5:
            results.append('Sandal')
        elif pca_df['label'][i] == 6:
            results.append('Shirt')
        elif pca_df['label'][i] == 7:
            results.append('Sneaker')
        elif pca_df['label'][i] == 8:
            results.append('Bag')
        elif pca_df['label'][i] == 9:
            results.append('Ankle boot')
        else:
            print("The dataset contains an unexpected label {}".format(pca_df['label'][i]))

    # Create a new column named result which has all meaningful results
    pca_df['result'] = results

    N = 10000
    pca_df_subset = pca_df.loc[rand[:N], :].copy()
    data_subset = pca_df_subset[features].values
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_subset)
    pca_df_subset['First Dimension'] = pca_result[:, 0]
    pca_df_subset['Second Dimension'] = pca_result[:, 1]
    pca_df_subset['Third Dimension'] = pca_result[:, 2]
    print('Explained variation in each principal component: {}'.format(pca.explained_variance_ratio_))
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE finished! Time elapsed: {} seconds'.format(time.time() - time_start))

    pca_df_subset['t-SNE First Dimension'] = tsne_results[:, 0]
    pca_df_subset['t-SNE Second Dimension'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="t-SNE First Dimension", y="t-SNE Second Dimension",
        hue="result",
        hue_order=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot'],
        palette=sns.color_palette("hls", 10),
        data=pca_df_subset,
        legend="full",
        alpha=0.3
    )
    plt.title("Scatter Plot of 10000 samples (Aseel Sabri and Kareem Afaneh)")
    plt.show()


def KNNOneManhattan(x_train, y_train, x_val, y_val):
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
    # Train the model using the training sets
    knn.fit(x_train, y_train)
    pred = knn.score(x_val, y_val)
    print("KNN with k = 1 using Manhattan Distance: " + str(pred))
    write_On_file("KNN", "KNN with k = 1 using Manhattan Distance: " + str(pred) + "\n")
    store_models("KNNOneManhattan.pkl", knn)


def KNNThreeManhattan(x_train, y_train, x_val, y_val):
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
    # Train the model using the training sets
    knn.fit(x_train, y_train)
    pred = knn.score(x_val, y_val)
    print("KNN with k = 3 using Manhattan Distance: " + str(pred))
    write_On_file("KNN", "KNN with k = 3 using Manhattan Distance: " + str(pred) + "\n")
    store_models("KNNThreeManhattan.pkl", knn)


def KNNOneEuclidean(x_train, y_train, x_val, y_val):
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    # Train the model using the training sets
    knn.fit(x_train, y_train)
    pred = knn.score(x_val, y_val)
    print("KNN with k = 1 using Euclidean Distance: " + str(pred))
    write_On_file("KNN", "KNN with k = 1 using Euclidean Distance: " + str(pred) + "\n")
    store_models("KNNOneEuclidean.pkl", knn)


def KNNThreeEuclidean(x_train, y_train, x_val, y_val):
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    # Train the model using the training sets
    knn.fit(x_train, y_train)
    pred = knn.score(x_val, y_val)
    print("KNN with k = 3 using Euclidean Distance: " + str(pred))
    write_On_file("KNN", "KNN with k = 3 using Euclidean Distance: " + str(pred) + "\n")
    store_models("KNNThreeEuclidean.pkl", knn)


def KNN_BaselineModel(x_train, y_train, x_test, y_test):
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
    # Train the model using the training sets
    knn.fit(x_train, y_train)
    pred = knn.score(x_test, y_test)
    print("Accuracy of the Baseline Model (KNN with k = 3 using manhattan Distance): " + str(pred))
    write_On_file("KNN",
                  "\n\nAccuracy of the Baseline Model (KNN with k = 3 using manhattan Distance): " + str(pred) + "\n")
    store_models("KNN_BaselineModel.pkl", knn)


def model1_NeuralNetwork(x_train, y_train, x_val, y_val):
    NeuralNetwork1 = MLPClassifier(solver='adam', max_iter=500, alpha=1e-5, hidden_layer_sizes=(10,),
                                   random_state=1, tol=1e-3, n_iter_no_change=5)
    NeuralNetwork1.fit(x_train, y_train)
    pred = NeuralNetwork1.score(x_val, y_val)
    print("Accuracy of the Neural Network (MLP) with hidden layers 10: " + str(pred))
    store_models("NeuralNetwork1.pkl", NeuralNetwork1)

    NeuralNetwork2 = MLPClassifier(solver='adam', max_iter=500, alpha=1e-5, hidden_layer_sizes=(20,),
                                   random_state=1, tol=1e-3, n_iter_no_change=5)
    NeuralNetwork2.fit(x_train, y_train)
    pred = NeuralNetwork2.score(x_val, y_val)
    print("Accuracy of the Neural Network (MLP) with hidden layers 20: " + str(pred))
    store_models("NeuralNetwork2.pkl", NeuralNetwork2)

    NeuralNetwork3 = MLPClassifier(solver='adam', max_iter=500, alpha=1e-5, hidden_layer_sizes=(30,),
                                   random_state=1, tol=1e-3, n_iter_no_change=5)
    NeuralNetwork3.fit(x_train, y_train)
    pred = NeuralNetwork3.score(x_val, y_val)
    print("Accuracy of the Neural Network (MLP) with hidden layers 30: " + str(pred))
    store_models("NeuralNetwork2.pkl", NeuralNetwork3)


def model2_RandomForest(x_train, y_train, x_val, y_val):
    # training random Forest
    randomF = RandomForestClassifier(n_estimators=100, random_state=123)
    randomF.fit(x_train, y_train)
    pred = randomF.score(x_val, y_val)
    print("Random Forest Classifier  (max_features=sqrt): " + str(pred))
    write_On_file("Random_Forest", "Random Forest Classifier (max_features=sqrt): " + str(pred))
    store_models("randomForest.pkl", randomF)

    randomF = RandomForestClassifier(n_estimators=100, random_state=123, max_features='log2')
    randomF.fit(x_train, y_train)
    pred = randomF.score(x_val, y_val)
    print("Random Forest Classifier (max_features=log2): " + str(pred))
    write_On_file("Random_Forest", "Random Forest Classifier (max_features=log2): " + str(pred) + "\n")
    store_models("randomForest.pkl", randomF)

    randomF = RandomForestClassifier(n_estimators=100, random_state=123, max_features=0.6)
    randomF.fit(x_train, y_train)
    pred = randomF.score(x_val, y_val)
    print("Random Forest Classifier (max_features=0.6): " + str(pred))
    write_On_file("Random_Forest", "Random Forest Classifier (max_features=0.6): " + str(pred) + "\n")
    store_models("randomForest.pkl", randomF)


def model3_CNN(id, x_train, y_train, x_val, y_val, batch_size, epochs=20, num_classes=10):
    CNN = models.Sequential()
    CNN.add(layers.Conv2D(batch_size / 4, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    CNN.add(layers.MaxPooling2D((2, 2)))
    CNN.add(layers.Conv2D(batch_size / 2, (3, 3), activation='relu'))
    CNN.add(layers.MaxPooling2D((2, 2)))
    CNN.add(layers.Conv2D(batch_size, (3, 3), activation='relu'))
    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(batch_size, activation='relu'))
    CNN.add(layers.Dense(num_classes))
    CNN.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = CNN.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))
    test_loss, test_acc = CNN.evaluate(x_val, y_val)
    print(
        "Accuracy of the Convolutional Neural Network (CNN) with batchSize " + str(batch_size) + " : " + str(test_acc))
    write_On_file("CNN",
                  "Accuracy of the Convolutional Neural Network (CNN) with batchSize " + str(batch_size) + " : " + str(
                      test_acc) + "\n")
    store_models("CNN" + id + ".pkl", CNN)


def start():
    x_train, y_train, x_test, y_test, class_names = read_data()
    # x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_train = x_train / 255
    x_test = x_test / 255

    x_train_splitted, x_validation, y_train_splitted, y_validation = train_test_split(x_train, y_train,
                                                                                      random_state=104,
                                                                                      test_size=1 / 3, shuffle=True)

    # This part for finding the best baseline model between knn = 1 and 3, and with two different distances
    # st = time.time()
    # KNNOneManhattan(x_train_splitted, y_train, x_validation, y_validation)
    # KNNThreeManhattan(x_train_splitted, y_train, x_validation, y_validation)
    # KNNOneEuclidean(x_train_splitted, y_train, x_validation, y_validation)
    # KNNThreeEuclidean(x_train_splitted, y_train, x_validation, y_validation)
    # print('Execution time:', time.time() - st, 'seconds')

    # this part is to train the best baseline model chosen from the previous part which is knn with k=3
    # and using Manhattan distance
    # st = time.time()
    # KNN_BaselineModel(x_train, y_train, x_test, y_test)
    # print('Execution time:', time.time() - st, 'seconds')
    # KNN_BaselineModel = restore_models("KNN_BaselineModel.pkl")
    # f1 = f1_score(y_test, KNN_BaselineModel.predict(x_test), average=None)
    # print(f1)

    # This part discuss the Neural Network
    # st = time.time()
    # model1_NeuralNetwork(x_train_splitted, y_train, x_validation, y_validation)
    # print('Execution time:', time.time() - st, 'seconds')

    # st = time.time()
    # model2_RandomForest(x_train_splitted, y_train, x_validation, y_validation)
    # print('Execution time:', time.time() - st, 'seconds')

    # x_train_shaped = x_train_splitted.reshape(-1, 28, 28, 1)
    # x_validation_shaped = x_validation.reshape(-1, 28, 28, 1)
    # st = time.time()
    # model3_CNN(1, x_train_shaped, y_train, x_validation_shaped, y_validation, 32, 20, 10)
    # model3_CNN(2, x_train_shaped, y_train, x_validation_shaped, y_validation, 64, 20, 10)
    # model3_CNN(3, x_train_shaped, y_train, x_validation_shaped, y_validation, 128, 20, 10)
    # print('Execution time:', time.time() - st, 'seconds')

    # x_train_splitted = x_train_splitted.reshape(40000, 28, 28)
    # plt.figure()
    # plt.imshow(x_train_splitted[6], cmap=plt.cm.binary)
    # print(class_names[y_train[6]])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    # KNN_BaselineModel(x_train, y_train, x_test, y_test)
    KNN_Model = restore_models("KNN_BaselineModel.pkl")
    y_pred = KNN_Model.predict(x_test)
    indices = (y_pred != y_test)
    misclassified_samples = x_test[indices]
    actual_label = y_test[indices]
    pred_label = y_pred[indices]
    plot_misclassified(pred_label, misclassified_samples, actual_label, class_names)


def plot_misclassified(pred_label, misclassified_samples, actual_label, class_names):
    class_indices = []
    for i in range(10):
        class_indices.append(np.where(actual_label == i))

    k = 0
    for class_index in class_indices:
        # print(type(class_index))
        images = misclassified_samples[class_index]
        false_label = pred_label[class_index]
        plt.figure(figsize=(15, 12))
        plt.suptitle(class_names[k])
        for i in range(20):
            plt.subplot(4, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i].reshape(28, 28), cmap=plt.cm.binary)
            plt.xlabel(class_names[false_label[i]])
        plt.savefig(class_names[k])
        k += 1


if __name__ == '__main__':
    start()
