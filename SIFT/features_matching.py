import cv2

# read the images
img1 = cv2.imread('images_1/1/1.jpg')
img2 = cv2.imread('images_1/1/2.jpg')

# Convert images to grayscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Create SIFT object
sift = cv2.SIFT_create()

# Detect SIFT features in both images
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
print('End descriptors')
# Create feature matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# Match descriptors of both images
matches = bf.match(des1,des2)
print('End matches')
# Sort matches by distance
matches = sorted(matches, key = lambda x:x.distance)
print('Start drawing')
# Draw first 50 matches
matched_img = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags=2)
print('Start showing')
# Show the image
cv2.imshow('image', matched_img)
print('Start saving')
# Save the image
cv2.imwrite('images_1/1/features_matching.jpg', matched_img)
print("Waiting...")
cv2.waitKey(0)
cv2.destroyAllWindows()