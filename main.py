import mnist_reader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import random



def ScotterPlotOfTrasiningSamples(X_train,y_train):
    y_train = y_train.reshape(60000, 1)
    input_data = X_train / 255
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

def KNNOneManhattan(x_Val,y_Val,x_test,y_test):
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
    # Train the model using the training sets
    knn.fit(x_Val, y_Val)
    pred = knn.score(x_test,y_test)
    print("KNN with k = 1 using Manhattan Distance: " + str(pred))

def KNNThreeManhattan(x_Val,y_Val,x_test,y_test):
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
    # Train the model using the training sets
    knn.fit(x_Val, y_Val)
    pred = knn.score(x_test, y_test)
    print("KNN with k = 3 using Manhattan Distance: " + str(pred))

def KNNOneEuclidean(x_Val,y_Val,x_test,y_test):
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    # Train the model using the training sets
    knn.fit(x_Val, y_Val)
    pred = knn.score(x_test, y_test)
    print("KNN with k = 1 using Euclidean Distance: " + str(pred))

def KNNThreeEuclidean(x_Val,y_Val,x_test,y_test):
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    # Train the model using the training sets
    knn.fit(x_Val, y_Val)
    pred = knn.score(x_test, y_test)
    print("KNN with k = 3 using Euclidean Distance: " + str(pred))

def start():
    X_train, y_train = mnist_reader.load_mnist('', kind='train')
    X_test, y_test = mnist_reader.load_mnist('', kind='t10k')
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_train, X_validation,y_train, y_validation = train_test_split(X_train, y_train,random_state=104,test_size=1/3,shuffle=True)
    st = time.time()
    KNNOneManhattan(X_validation,y_validation,X_test,y_test)
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    #KNNThreeManhattan(X_validation,y_validation,X_test,y_test)
    #KNNOneEuclidean(X_validation,y_validation,X_test,y_test)
    #KNNThreeEuclidean(X_validation,y_validation,X_test,y_test)

    # X_train= X_train.reshape(40000,28,28)
    # plt.figure()
    # plt.imshow(X_train[6], cmap=plt.cm.binary)
    # print(class_names[y_train[6]])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    # ScotterPlotOfTrasiningSamples(X_train,y_train)


if __name__ == '__main__':
    start()



