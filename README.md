# Image Segmentation using K-means

This repository contains two scripts named `xxx_face(s).py` which implement an image segmentation algorithm using the K-means clustering technique. The purpose of this project is to identify the faces in an image and convert it to an output image with a bounding box around the detected faces.

  - Single Face
    <p>
      <img src="https://user-images.githubusercontent.com/114009025/230433223-5abbd60c-0b3c-48e2-835e-c253a499179e.jpg" width="15%" height="15%"/>
        &rarr;
      <img src="https://user-images.githubusercontent.com/114009025/230434864-bfd5d378-615b-4fff-b1c6-e3b2c43ad7eb.jpg" width="15%" height="15%"/> 
        &rarr;
      <img src="https://user-images.githubusercontent.com/114009025/230434902-ac042a92-89c6-4986-9e89-ccf3d53678ab.jpg" width="15%" height="15%"/>
    </p>

  - Multiple Faces
    <p>
      <img src="https://user-images.githubusercontent.com/114009025/230434969-297d5bb3-c531-4ac4-8b86-c936ddfcc95c.jpg" width="25%" height="25%"/>
        &rarr;
      <img src="https://user-images.githubusercontent.com/114009025/230435017-7da2db2c-d2b6-44b4-a258-ec3387d849e4.jpg" width="25%" height="25%"/>
    </p>

## Logic of the Flow

After reading the image in as an array, the script converts the image to the Lab color space and extracts only the blue-yellow component. The extracted values are reshaped into a 2-dimensional array and fed into the K-means clustering algorithm with k=4, which is chosen using the elbow method. The resulting clustering is filtered to include only the points related to the face, based on specified RGB constraints. Then, the script calculates the median and standard deviation of the x, y indices for the facial-related points in order to construct a rectangle around the face.

## Key Variables/Parameters

- `img_filePath`: The path to the input image file.
- `img_color`: The original image array in BGR format.
- `img_RGB`: The image array in RGB format.
- `img_Lab`: The image array in Lab color space.
- `img_BY_2d`: The blue-yellow component from img_Lab reshaped to a 2-dimensional array.
- `kmeans`: The fitted K-means model.
- `RGB_mean`: The mean values for each of the RGB channels within each cluster.
- `h_axis`: The horizontal coordinates within the facial-related cluster.
- `v_axis`: The vertical coordinates within the facial-related cluster.
- `center_x/y`: The median of h_axis and v_axis, which serves as the center for the rectangle.
- `half_w/h`: 1.3 * std of h_axis and v_axis, which serves as half the width and height of the rectangle.
- `(x,y)/(x1, y1)`: The upper left/lower right corner of the rectangle.
- `img_savePath`: the path to save image.
