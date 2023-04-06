import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def kmeans(imgPath, imgFilename, savedImgPath, savedImgFilename, k):
    """
        parameters:
        imgPath: the path of the image folder. (relative path)
        imgFilename: the name of the image file
        savedImgPath: the path of the folder to save the image
        savedImgFilename: the name of the output image
        k: the number of clusters of the k-means function
        function: using k-means to segment the image and save the result to an image with bounding boxes
    """
    # Read image
    img_filePath = os.path.join(imgPath, imgFilename)
    img_color = cv2.imread(img_filePath)
    img_RGB = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Opencv uses BGR, convert to RGB

    # Data Processing
    img_Lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)  # convert the img to Lab color space
    img_2d = img_Lab[:, :, 1:].reshape(-1, 2)  # extract only the green-red and blue-yellow component and reshape to 2d
    xy = [[i % img_color.shape[1], i // img_color.shape[1]] for i in range(len(img_2d))]  # get x/y labels for each pts
    img_2d_xy = np.hstack([img_2d, np.array(xy)])  # horizontally stack GR-BY-xy to form the training set

    scaler = StandardScaler()  # Standardize training data [img_2d_xy]
    img_2d_xy = scaler.fit_transform(img_2d_xy)

    # kmeans clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_2d_xy)

    # Find the cluster label for face using specified RGB constraints
    for label in range(k):
        RGB_mean = np.array([img_RGB.reshape(-1, 3)[i].tolist() for i in range(0, len(kmeans.labels_)) if
                             kmeans.labels_[i] == label]).mean(axis=0)
        if (RGB_mean[0] > 160) and (RGB_mean[1] < 160) and (RGB_mean[2] < 150):
            face_cluster_label = label
            break

    # ———————————— Determine the h_axis value range for faces ————————————
    h_axis = [i % img_color.shape[1] for i in range(len(img_2d_xy)) if
              kmeans.labels_[i] == face_cluster_label]  # get the horizontal axis values for the face cluster

    hist, edges = np.histogram(
        h_axis,
        bins=15,
        density=False)

    face_start_bin_num = []
    face_end_bin_num = []
    enter_face_range = False

    # find the range for the four peaks horizontally (exclude values w/ freq less than 100)
    for i in range(len(hist)):
        if enter_face_range and hist[i] < 100:
            enter_face_range = False
            face_end_bin_num.append(i - 1)
        elif not enter_face_range and hist[i] > 100:
            enter_face_range = True
            face_start_bin_num.append(i)
    face_end_bin_num.append(len(hist) - 1)  # append the last h_axis value for face

    face_start_h_val = [int(edges[i]) for i in face_start_bin_num]  # 4 start h_values for faces
    face_end_h_val = [int(edges[i + 1]) for i in face_end_bin_num]  # 4 end h_values for faces

    # ———————————— Determine the v_axis value range for faces ————————————
    face_start_v_val = []
    face_end_v_val = []

    for face_num in range(4):
        # look into the distribution of vertical axis values for each paris of h_vals seperately
        face_v_val = [i // img_color.shape[1] for i in range(len(img_2d_xy)) if
                      kmeans.labels_[i] == face_cluster_label and i % img_color.shape[1] in range(
                          face_start_h_val[face_num],
                          face_end_h_val[face_num])]
        hist, edges = np.histogram(face_v_val, density=False)

        face_v_range = np.array(
            [edges[i: i + 2] for i in range(len(hist)) if hist[i] > 100])  # filter out bins w/ freq less than 100
        face_start_v_val.append(int(np.array(face_v_range).min()))  # 4 start v_values for faces
        face_end_v_val.append(int(np.array(face_v_range).max()))  # 4 end v_values for faces

    # Add the four rectangular boxes onto the image
    for face_num in range(4):
        img_color_rect = cv2.rectangle(np.ascontiguousarray(img_color, dtype=np.uint8),
                                       (face_start_h_val[face_num], face_start_v_val[face_num]),
                                       (face_end_h_val[face_num], face_end_v_val[face_num]), (0, 255, 0), 2)

    # plt.imshow(img_color_rect)
    # plt.show()

    # Save image
    img_savePath = os.path.join(savedImgPath, savedImgFilename)
    cv2.imwrite(img_savePath, img_color_rect)


if __name__ == "__main__":
    imgPath = ""
    imgFilename = "faces.jpg"
    savedImgPath = r''
    savedImgFilename = "faces_detected_4faces.jpg"
    k = 15
    kmeans(imgPath, imgFilename, savedImgPath, savedImgFilename, k)
