import numpy as np
import cv2
from matplotlib import pyplot as plt


def main():
    path = 'Figureprint matching'
    
    # Fingerprint images same
    same1 = cv2.imread(str(path) + '/same5_1.tif')                      # read/load the image
    same2 = cv2.imread(str(path) + '/same5_2.tif')                      # read/load the image

    # Fingerprint images different
    different1 = cv2.imread(str(path) + '/dif3_1.tif')                  # read/load the image
    different2 = cv2.imread(str(path) + '/dif3_2.tif')                  # read/load the image

    # UIA images
    UIA1 = cv2.imread(str(path) + '/UIA_1.png')                         # read/load the image
    UIA2 = cv2.imread(str(path) + '/UIA_2.png')                         # read/load the image

    # ORB
    result_orb = orb_match(same1, same2)

    plt.figure()
    plt.imshow(result_orb)
    plt.title('ORB Matching')
    plt.axis('off')
    plt.show()
    cv2.imwrite(str(path) + '/result_orb.png', result_orb)

    # SIFT
    result_sift = sift_match(same1, same2)
    plt.figure()
    plt.imshow(result_sift)
    plt.title('SIFT Matching')
    plt.axis('off')
    plt.show()
    cv2.imwrite(str(path) + '/result_sift.png', result_sift)      

    # BFMatcher
    result_bf = brute_force_match(same1, same2)
    plt.figure()
    plt.imshow(result_bf)
    plt.title('Brute Force Matching')
    plt.axis('off')
    plt.show()
    cv2.imwrite(str(path) + '/result_bf.png', result_bf)

    # FLANN
    result_flann = flann_match(same1, same2)
    plt.figure()
    plt.imshow(result_flann)
    plt.title('FLANN Matching')
    plt.axis('off')
    plt.show()
    cv2.imwrite(str(path) + '/result_flann.png', result_flann)




# Method: Feature Matching Operations

# ORB (Oriented FAST and Rotated BRIEF) 
def orb_match(same1, same2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(same1, None)
    kp2, des2 = orb.detectAndCompute(same2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(same1, kp1, same2, kp2, matches[:30], None, flags=2)
    return img_matches

# SIFT (Scale-Invariant Feature Transform)
def sift_match(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    return img_matches

# Brute Force Matcher (BFMatcher)
def brute_force_match(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)
    return img_matches

# FLANN (Fast Library for Approximate Nearest Neighbors)
def flann_match(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    return img_matches

main()

