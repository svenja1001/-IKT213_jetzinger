import numpy as np
import cv2
from matplotlib import pyplot as plt

def main():
    path = 'assignment_3'
    img = cv2.imread(str(path) + '/lambo.png')                     # read/load the image
    
    sobel_edge_detection(img)



# Sobel edge detection
def sobel_edge_detection(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)                        # ksize=(3,3), sigmaX=0
    # Sobel edge detection
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=1)            # Sobel in x direction with dx=1, dy=0
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=1)            # Sobel in y direction with dx=0, dy=1
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)                # Combine the two directions (x and y)
    # Show image
    plt.imshow(sobel_combined)
    plt.title('Sobel Edge Detection')
    plt.axis('off')




main()