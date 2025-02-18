from os.path import isdir

import cv2
import os

mainDir = 'images_4'
directories = os.listdir(mainDir)
i = 0
for directory in directories:
    path = os.listdir(os.path.join(mainDir, directory))

    images = []
    imagePaths = path
    for image in imagePaths:
        # Load images with open cv and put into the list images
        loadedIamge = cv2.imread(os.path.join(mainDir, directory, image))
        images.append(loadedIamge)

    # Create the panorama using open cv
    imageStitcher = cv2.Stitcher.create()
    (status, panorama) = imageStitcher.stitch(images)
    if status == cv2.STITCHER_OK:
        print('Successfully stitched image')
        cv2.imwrite(os.path.join(mainDir, f'panorama_{i}.png'), panorama)
        cv2.imshow('image', panorama)
        i += 1
    else:
        print(f'Failed to stitch image. Status {status}')
cv2.waitKey(0)
