import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def kmeans(imgPath, imgFilename, savedImgPath, savedImgFilename, k):
    """
        parameters:
        imgPath: the path of the image folder. Please use relative path
        imgFilename: the name of the image file
        savedImgPath: the path of the folder you will save the image
        savedImgFilename: the name of the output image
        k: the number of clusters of the k-means function
        function: using k-means to segment the image and save the result to an image with a bounding box
    """
    # Read image
    img_filePath = os.path.join(imgPath, imgFilename)
    img_color = cv2.imread(img_filePath)

    # Image data conversion
    img_RGB = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Opencv uses BGR, convert to RGB
    img_Lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)  # convert the img to Lab color space
    img_BY_2d = img_Lab[:, :, -1].reshape(-1, 1)  # extract only the blue-yellow component and reshape to 2d

    # kmeans clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_BY_2d)

    # Find the cluster label for face using specified RGB constraints
    for label in range(k):
        RGB_mean = np.array([img_RGB.reshape(-1, 3)[i].tolist() for i in range(0, len(kmeans.labels_)) if
                             kmeans.labels_[i] == label]).mean(axis=0)
        if (RGB_mean[0] > 160) and (RGB_mean[1] < 160) and (RGB_mean[2] < 150):
            face_cluster_label = label
            break

    # Determine the x/y range for faces
    h_axis = [i % img_color.shape[1] for i in range(len(img_BY_2d)) if
              kmeans.labels_[i] == face_cluster_label]  # horizontal coordinates where face appear
    v_axis = [i // img_color.shape[1] for i in range(len(img_BY_2d)) if
              kmeans.labels_[i] == face_cluster_label]  # vertical coordinates where face appear
    center_x = np.median(h_axis)
    center_y = np.median(v_axis)
    half_w = 1.3 * np.std(h_axis)  # half width for rectangle
    half_h = 1.3 * np.std(v_axis)  # half height for rectangle

    x = int(center_x - half_w)  # upper left corner (x,y)
    x1 = int(center_x + half_w)
    y = int(center_y - half_h)  # lower right corner (x1,y1)
    y1 = int(center_y + half_h)
    # Add the rectangular box onto the image
    img_color_rect = cv2.rectangle(np.ascontiguousarray(img_color, dtype=np.uint8), (x, y), (x1, y1), (0, 255, 0), 2)

    # plt.imshow(img_color_rect)
    # plt.show()

    # Save image
    img_savePath = os.path.join(savedImgPath, savedImgFilename)
    cv2.imwrite(img_savePath, img_color_rect)


if __name__ == "__main__":
    imgPath = ""
    imgFilename = "face_d2.jpg"
    savedImgPath = r''
    savedImgFilename = "face_d2_face.jpg"
    k = 4
    kmeans(imgPath, imgFilename, savedImgPath, savedImgFilename, k)
