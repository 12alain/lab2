import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to load and stitch images
def stitch_images(image1, image2):
    # Initialize the SIFT detector
    sift = cv2.SIFT_create(nfeatures=None)

    # Detect keypoints and descriptors in both images
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    # Use FLANN-based matcher to match descriptors
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # If there are not enough good matches, return None
    if len(good_matches) < 4:
        print("Not enough matches found.")
        return None

    # Extract location of good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # Get the dimensions of the first image

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Warp the first image to the second image using the homography matrix
    result = cv2.warpPerspective(image1, H, (w1+w2, h1+h2))

    # Place the second image onto the warped first image
    result[0:h2, 0:w2] = image2

    return result


mainDir = 'images_1'
directories = os.listdir(mainDir)
i = 0
for directory in directories:
    path = os.listdir(os.path.join(mainDir, directory))

    images = []
    imagePaths = path
    for image in imagePaths:
        # Load images with open cv and put into the list images
        loadedImage = cv2.imread(os.path.join(mainDir, directory, image))
        images.append(loadedImage)
    start = 0
    while start + 1 < len(images):

        stitched_image = stitch_images(images[start], images[start + 1])

        # Show the result
        if stitched_image is not None:
            # Save image
            cv2.imwrite(os.path.join(mainDir, directory, f'stitched_{i}.jpg'), stitched_image)
            i +=1
            plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        else:
            print("Stitching failed.")

        start = start + 2
