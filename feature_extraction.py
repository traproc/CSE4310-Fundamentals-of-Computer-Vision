# Hien Dao 1001912046
# CSE 4310 - Fundamentals of Computer Vision

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from skimage.color import rgb2gray
from skimage.feature import SIFT
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances

def keypoint_matching(desc1,desc2):
    distances = euclidean_distances(desc1, desc2)
    dst = np.arange(len(desc1))
    src = np.argmin(distances, axis=1)

    # Cross-checks keypoints
    check_matches = np.argmin(distances, axis=0)
    mask = dst == check_matches[src]
    dst = dst[mask]
    src = src[mask]

    # Construct matches array
    matches = np.column_stack((dst, src))
    return np.array(matches)

def plot_keypoint_matching(matches, img1, img2, kp1, kp2):
    dst_best = kp1[matches[:,0]][:, ::-1]
    src_best = kp2[matches[:,1]][:, ::-1]

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(img1)
    ax2.imshow(img2)

    for i in range(src_best.shape[0]):
        coordB = [dst_best[i, 0], dst_best[i, 1]]
        coordA = [src_best[i, 0], src_best[i, 1]]
        con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data",
                            axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)
        ax1.plot(dst_best[i, 0], dst_best[i, 1], 'ro')
        ax2.plot(src_best[i, 0], src_best[i, 1], 'ro')
    plt.show()

def feature_process_sift(data, target):
    # Convert the input data to a numpy array. It is saved in row-major order where the first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
    cifar_rgb = np.array(data, dtype='uint8')

    # Reshape the data to (num_images, height, width, num_channels)
    cifar_rgb = cifar_rgb.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Convert the images to grayscale
    cifar_gray = rgb2gray(cifar_rgb)

    # Extract SIFT features per class
    sift = SIFT()
    sift_features = []
    y_features = []

    for idx in tqdm(range(cifar_gray.shape[0]), desc="Processing images"):
        try:
            sift.detect_and_extract(cifar_gray[idx])
            sift_features.append(sift.descriptors)
            y_features.append(target[idx]) # Only stores the label if the SIFT features are successfully extracted
        except:
            pass

    return sift.keypoints, sift_features, y_features

def create_bag_of_visual_words(features, k):
    # Convert the list of SIFT features to a numpy array
    features_np = np.concatenate(features)
    # Create a KMeans model to cluster the SIFT features
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Fit the KMeans model to the SIFT features
    kmeans.fit(features_np)

    # Build a histogram of the cluster centers for each image using the features already extracted
    image_histograms = []

    for feature in tqdm(features, desc="Building histograms"):
        # Predict the closest cluster for each feature
        clusters = kmeans.predict(feature)
        # Build a histogram of the clusters
        histogram, _ = np.histogram(clusters, bins=k, range=(0, k))
        image_histograms.append(histogram)

    # Convert the list of histograms to a numpy array
    image_histograms_np = np.array(image_histograms)

    # Adjust frequency using TF-IDF

    # Create a TfidfTransformer
    tfidf = TfidfTransformer()

    # Fit the TfidfTransformer to the histogram data
    tfidf.fit(image_histograms_np)

    # Transform the histogram data using the trained TfidfTransformer
    image_histograms_tfidf = tfidf.transform(image_histograms_np)

    return image_histograms_tfidf

if __name__ == "__main__":
    # Load the pre-split data
    data = np.load("cifar10.npz", allow_pickle=True)

    # Extract features from the training data
    kp_train, desc_train, y_features_train = feature_process_sift(data["X_train"], data["y_train"])

    # Extract features from the testing data
    kp_test, desc_test, y_features_test = feature_process_sift(data["X_test"], data["y_test"])

    # Save the extracted features to a file
    k_sift = 100
    sift_histogram_train = create_bag_of_visual_words(desc_train, k_sift)
    sift_histogram_test = create_bag_of_visual_words(desc_test, k_sift)
    X_train, X_test, y_train, y_test = train_test_split(sift_histogram_train, np.array(y_features_train, dtype=int), test_size=0.2, random_state=42)

    # Save the dictionary to a file
    sift_data = {
        "X_train": X_train.toarray(),
        "X_test": X_test.toarray(),
        "y_train": y_train,
        "y_test": y_test
    }

    np.savez("sift_histogram.npz", **sift_data)
    
