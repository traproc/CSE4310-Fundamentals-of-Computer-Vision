# Hien Dao 1001912046
# CSE 4310 - Fundamentals of Computer Vision

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import skimage.io as io
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import SIFT
from skimage.transform import AffineTransform, ProjectiveTransform, SimilarityTransform, warp
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

def compute_affine_transform(kp1, kp2, matches):
    src = np.array(kp2[matches[:,1]])
    dst = np.array(kp1[matches[:,0]])
    
    # Construct the coefficient matrix
    A = []
    b = []

    for i in range(len(matches)):
        x1, y1 = src[i]
        x2, y2 = dst[i]
        A.append([x1, y1, 1, 0, 0, 0])
        A.append([0, 0, 0, x1, y1, 1])
        b.append(x2)
        b.append(y2)
    A = np.asarray(A)
    b = np.asarray(b)

    # Solve the system of linear equations
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Reshape the parameters into a 2x3 affine matrix and add [0,0,1] to bottom of matrix
    H = np.reshape(x, (2, 3))
    H = np.vstack([H, [0,0,1]])

    return H

def compute_projective_transform(kp1, kp2, matches):
    dest = np.array(kp1[matches[:,0]])
    src = np.array(kp2[matches[:,1]])

    # Construct the coefficient matrix
    A = []

    for i in range(len(matches)):
        x1, y1 = src[i]
        x2, y2 = dest[i]
        A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2])
        A.append([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2])

    # Solve the system of linear equations
    _, _, V = np.linalg.svd(A)

    # Reshape the parameters into a 3x3 projective matrix
    H = np.reshape(V[-1], (3, 3))
    
    H /= H[2,2]

    return H

'''
RANSAC
Given:
    data - A set of observations.
    model - A model to explain the observed data points.
    n - The minimum number of data points required to estimate the model parameters.
    k - The maximum number of iterations allowed in the algorithm.
    t - A threshold value to determine data points that are fit well by the model (inlier).
    d - The number of close data points (inliers) required to assert that the model fits well to the data.

Return:
    bestFit - The model parameters which may best fit the data (or null if no good model is found).
    bestInliers - the matches with the lowest error
'''

def ransac(kp1, kp2, matches, k, n, t, d):
    iterations = 0
    bestFit = None
    bestInliers = []
    kp1 = kp1[:,::-1]
    kp2 = kp2[:,::-1]
    
    while iterations < k:
        # Generate a random sample size within the range [n, len(matches)]
        sample_size = np.random.randint(n, len(matches) + 1)
        maybeInliers = random.sample(range(len(matches)), sample_size)

        # Model parameters fitted to maybeInliers using either affine or projective transformatino
        #model = compute_affine_transform(kp1, kp2, matches[maybeInliers])
        model = compute_projective_transform(kp1, kp2, matches[maybeInliers])
        confirmedInliers = []
        for i, point in enumerate(matches):
            src_point = np.append(kp2[point[1]], 1)  # Homogeneous coordinates 
            dst_point = np.append(kp1[point[0]], 1)  # Homogeneous coordinates
            transform = np.dot(model, src_point)
            transform /= transform[2] # Perspective divide
            error = np.linalg.norm(dst_point-transform)
            if error < t: #point fits model with an error smaller than < t then
                confirmedInliers.append(i)
        if len(confirmedInliers) > d:
            # This implies that we may have found a good model.
            # Now test how good it is.
            if len(confirmedInliers) > len(bestInliers):
                bestFit = model
                bestInliers = confirmedInliers
        iterations += 1

    return bestFit, bestInliers

def stitch_images(src_img, dst_img, V):
    # Transform the corners of img1 by the inverse of the best fit model
    H = ProjectiveTransform(V)
    #H = AffineTransform(V)
    rows, cols = dst_GRAY.shape
    corners = np.array([
        [0, 0],
        [cols, 0],
        [0, rows],
        [cols, rows]
    ])

    corners_proj = H(corners)

    all_corners = np.vstack((corners_proj[:, :2], corners[:, :2]))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = (corner_max - corner_min)
    output_shape = np.ceil(output_shape[::-1]).astype(int)
    print(output_shape)

    offset = SimilarityTransform(translation=-corner_min)
    dst_warped = warp(dst_img, offset.inverse, output_shape=output_shape)

    tf_img = warp(src_img, np.linalg.inv(H + offset), output_shape=output_shape)

    # Combine the images
    foreground_pixels = tf_img[tf_img > 0]
    dst_warped[tf_img > 0] = tf_img[tf_img > 0]

    # Plot the result
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(dst_warped)
    plt.show()

    return dst_warped

if __name__ == "__main__":
    #image_path = ["data/campus_004.jpg","data/campus_003.jpg","data/campus_002.jpg","data/campus_001.jpg","data/campus_000.jpg"]
    image_path = ["data/Rainier1.png","data/Rainier2.png"]
    #image_path = ["data/yosemite1.jpg","data/yosemite2.jpg","data/yosemite3.jpg","data/yosemite4.jpg"]
    images = []
    for filename in image_path:
        img = io.imread(filename)
        if img.shape[2] == 4:
            img = rgba2rgb(img)
        images.append(img)
    img = images[0]
    for i in range(1,len(images)):
        dst_img = img
        src_img = images[i]
        dst_GRAY = rgb2gray(dst_img)
        src_GRAY = rgb2gray(src_img)
        
        # Keypoint detection using SIFT
        print("Calculating keypoint detection in images")
        detector1 = SIFT()
        detector2 = SIFT()
        detector1.detect_and_extract(dst_GRAY)
        detector2.detect_and_extract(src_GRAY)
        kp1 = detector1.keypoints
        desc1 = detector1.descriptors
        kp2 = detector2.keypoints
        desc2 = detector2.descriptors

        # Keypoint matching between two images
        print("Matching keypoints")
        matches = keypoint_matching(desc1, desc2)
        print(str(len(matches))+" matches found")

        # Shows all the matching keypoints
        plot_keypoint_matching(matches, dst_img, src_img, kp1, kp2)

        # RANSAC
        print("Removing outliers using RANSAC")
        H, inliers = ransac(kp1, kp2, matches, len(matches), 4, 0.5, 0)
        print(str(len(inliers))+" inliers detected")

        # Shows all the matching keypoints
        plot_keypoint_matching(matches[inliers], dst_img, src_img, kp1, kp2)

        # Stitch images together
        print("Stitching images together\n")
        stitched_img = stitch_images(src_img, dst_img, H)

        img = (stitched_img * 255).astype('uint8')

    
