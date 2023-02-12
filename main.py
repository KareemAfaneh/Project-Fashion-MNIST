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
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from skimage.feature import hog


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


def KNN(k, metric, x_train, y_train, x_test, y_test, baseline=False):
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    # Train the model using the training sets
    knn.fit(x_train, y_train)
    test_accuracy = knn.score(x_test, y_test)
    write_On_file("KNN", "KNN2 with k = " + str(k) + " using " + metric + " Distance: " + str(test_accuracy) + "\n")
    if baseline:
        f1 = f1_score(y_test, knn.predict(x_test), average='macro')
        training_accuracy = knn.score(x_train, y_train)
        write_On_file("KNN_BaselineModel2", "\n\nTraining Accuracy of the Baseline Model "
                                           "(KNN with k = 3 using manhattan Distance): " + str(
            training_accuracy) + "\n")
        write_On_file("KNN_BaselineModel2", "\n\nTesting Accuracy of the Baseline Model "
                                           "(KNN with k = 3 using manhattan Distance): " + str(test_accuracy) + "\n")
        write_On_file("KNN_BaselineModel2", "\n\nF1-Score of the Baseline Model "
                                           "(KNN with k = 3 using manhattan Distance): " + str(f1) + "\n")

        store_models("KNN_BaselineModel2.pkl", knn)


def model1_NeuralNetwork(x_train, y_train, x_test, y_test):
    # NeuralNetwork1 = MLPClassifier(solver='adam', max_iter=500, learning_rate_init=0.001,
    #                                random_state=1, tol=1e-3, n_iter_no_change=5)
    # NeuralNetwork1.fit(x_train, y_train)
    # pred = NeuralNetwork1.score(x_test, y_test)
    # print("Accuracy of the Neural Network (MLP) with learning_rate_init=0.001: " + str(pred))
    # # store_models("NeuralNetwork1.pkl", NeuralNetwork1)
    #
    # NeuralNetwork2 = MLPClassifier(solver='adam', max_iter=500, learning_rate_init=0.005,
    #                                random_state=1, tol=1e-3, n_iter_no_change=5)
    # NeuralNetwork2.fit(x_train, y_train)
    # pred = NeuralNetwork2.score(x_test, y_test)
    # print("Accuracy of the Neural Network (MLP) with learning_rate_init=0.005: " + str(pred))

    NeuralNetwork3 = MLPClassifier(solver='adam', max_iter=500, learning_rate_init=0.0005,
                                   random_state=1, tol=1e-3, n_iter_no_change=5)
    NeuralNetwork3.fit(x_train, y_train)
    pred = NeuralNetwork3.score(x_test, y_test)
    print("Accuracy of the Neural Network (MLP) with learning_rate_init=0.0005: " + str(pred))
    store_models("NeuralNetwork2.pkl", NeuralNetwork3)


def model2_RandomForest(x_train, y_train, x_test, y_test, max_features, best_model=False):
    randomF = RandomForestClassifier(n_estimators=100, random_state=123, max_features=max_features)
    randomF.fit(x_train, y_train)
    accuracy = randomF.score(x_test, y_test)
    write_On_file("Random_Forest",
                  "Accuracy of Random Forest Classifier (max_features=" + str(max_features) + "): " + str(
                      accuracy) + "\n")
    if best_model:
        f1 = f1_score(y_test, randomF.predict(x_test), average='macro')
        training_accuracy = randomF.score(x_train, y_train)
        write_On_file("Random_Forest",
                      "Accuracy of Random Forest Classifier (max_features=" + str(
                          max_features) + "): " + "\nFor Testing set of data: " + str(accuracy) + "\n" +
                      "For Training set of data: " + str(training_accuracy) + "\n")
        write_On_file("Random_Forest",
                      "\nF1-Score of Random Forest Classifier (max_features=" + str(max_features) + "): " + str(
                          f1) + "\n")
        store_models("randomForest.pkl", randomF)


def model3_CNN(x_train, y_train, x_val, y_val, x_test, y_test, batch_size, bestModel, epochs=20, num_classes=10):
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
    val_loss, val_acc = CNN.evaluate(x_val, y_val)
    write_On_file("CNN", "Accuracy of the Convolutional Neural Network (CNN) with batchSize " + str(batch_size) +
                  " : " + str(val_acc) + "\n")
    if bestModel:
        val_loss, val_acc = CNN.evaluate(x_val, y_val)
        labels = np.argmax(CNN.predict(x_test), axis=-1)
        f1 = f1_score(y_test, labels, average='macro')
        training_loss, training_acc = CNN.evaluate(x_train, y_train)
        test_loss, test_acc = CNN.evaluate(x_test, y_test)
        write_On_file("CNN", "Accuracy of the Convolutional Neural Network (CNN) with batchSize " + str(batch_size) +
                      "\nFor Testing set of data: " + str(test_acc) + "\n" +
                      "For Training set of data: " + str(training_acc) + "\n" +
                      "For Validation set of data: " + str(val_acc) + "\n")
        write_On_file("CNN", "F1-Score of the Convolutional Neural Network (CNN) with batchSize " + str(batch_size) +
                      " : " + str(f1) + "\n")
        store_models("CNN.pkl", CNN)


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


def start():
    x_train, y_train, x_test, y_test, class_names = read_data()
    # x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_train = x_train / 255
    x_test = x_test / 255

    x_train_splitted, x_validation, y_train_splitted, y_validation = train_test_split(x_train, y_train,
                                                                                      random_state=104, test_size=1 / 3,
                                                                                      shuffle=True)

    # This part for finding the best baseline model between knn = 1 and 3, and with two different distances
    # st = time.time()
    # KNN(1, "manhattan", x_train_splitted, y_train_splitted, x_validation, y_validation)
    # KNN(3, "manhattan", x_train_splitted, y_train_splitted, x_validation, y_validation)  # baseline
    # KNN(1, "euclidean", x_train_splitted, y_train_splitted, x_validation, y_validation)
    # KNN(3, "euclidean", x_train_splitted, y_train_splitted, x_validation, y_validation)

    # retraining the baseline model (k=3, manhatten) on the whole data
    # KNN(3, "manhattan", x_train, y_train, x_test, y_test, True)

    # This part discuss the Neural Network
    st = time.time()
    # model1_NeuralNetwork(x_train_splitted, y_train_splitted, x_validation, y_validation)
    # model1_NeuralNetwork(x_train, y_train, x_test, y_test)
    # NeuralNetwork = restore_models("NeuralNetwork.pkl")
    # print(NeuralNetwork.score(x_train, y_train))
    # print('Execution time:', time.time() - st, 'seconds')

    # # tuning the max_features parameter in random forest
    # model2_RandomForest(x_train_splitted, y_train_splitted, x_validation, y_validation, 'sqrt')  # best Model
    # model2_RandomForest(x_train_splitted, y_train_splitted, x_validation, y_validation, 'log2')
    # model2_RandomForest(x_train_splitted, y_train_splitted, x_validation, y_validation, 0.6)
    #
    # # retraining the best random forest model (max_features) on the whole data
    # model2_RandomForest(x_train, y_train, x_test, y_test, 'sqrt', True)

    # x_train_shaped = x_train_splitted.reshape(-1, 28, 28, 1)
    # x_validation_shaped = x_validation.reshape(-1, 28, 28, 1)
    # x_test_shaped = x_test.reshape(-1, 28, 28, 1)
    # model3_CNN(x_train_shaped, y_train_splitted, x_validation_shaped, y_validation, x_test_shaped, y_test, 32, False, 20, 10)
    # model3_CNN(x_train_shaped, y_train_splitted, x_validation_shaped, y_validation, x_test_shaped, y_test, 64, False, 20, 10)
    # model3_CNN(x_train_shaped, y_train_splitted, x_validation_shaped, y_validation, x_test_shaped, y_test, 128, True, 20, 10) # best model
    # print('Execution time:', time.time() - st, 'seconds')

    # x_train_splitted = x_train_splitted.reshape(40000, 28, 28)
    # plt.figure()
    # plt.imshow(x_train_splitted[6], cmap=plt.cm.binary)
    # print(class_names[y_train[6]])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    # KNN_BaselineModel(x_train, y_train, x_test, y_test)
    # KNN_Model = restore_models("KNN_BaselineModel.pkl")
    # y_pred = KNN_Model.predict(x_test)
    # indices = (y_pred != y_test)
    # misclassified_samples = x_test[indices]
    # actual_label = y_test[indices]
    # pred_label = y_pred[indices]
    # plot_misclassified(pred_label, misclassified_samples, actual_label, class_names)

    t1 = time.time()
    hog_features = []
    for image in x_train:
        fd = hog(image.reshape(28, 28), orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm='L2')
        hog_features.append(fd)

    x_train = np.array(hog_features)

    print(x_train.shape)

    hog_features = []
    for image in x_test:
        fd = hog(image.reshape(28, 28), orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm='L2')
        hog_features.append(fd)

    x_test = np.array(hog_features)

    KNN(3, "manhattan", x_train, y_train, x_test, y_test, True)

    # model1_NeuralNetwork(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    start()
