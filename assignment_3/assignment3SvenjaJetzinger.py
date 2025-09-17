import numpy as np
import cv2
from matplotlib import pyplot as plt

def main():
    path = 'assignment_3'
    img = cv2.imread(str(path) + '/lambo.png')                     # read/load the image
    
    sobel_edge_detection(img, path)

    # Threshold for canny edge detection
    threshold_1 = 50
    threshold_2 = 50
    canny_edge_detection(img, threshold_1, threshold_2, path)

    # Template match
    picture = cv2.imread(str(path) + '/shapes.png')                         # read/load the main image
    template = cv2.imread(str(path) + '/shapes_template.jpg')               # read/load the template image
    template_match(picture, template, path)

    



# Sobel edge detection
def sobel_edge_detection(img, path):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)                        # ksize=(3,3), sigmaX=0
    # Sobel edge detection
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=1)            # Sobel in x direction with dx=1, dy=0
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=1)            # Sobel in y direction with dx=0, dy=1
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)                # Combine the two directions (x and y)
    # Show image
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    plt.show()
    # Save image into the assignment_3 folder
    cv2.imwrite(str(path) + '/lambo_sobel_SvenjaJetzinger_A3.png', sobel_combined)


# Canny edge detection
def canny_edge_detection(img, threshold_1, threshold_2, path):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)                        # ksize=(3,3), sigmaX=0
    # Canny edge detection
    canny = cv2.Canny(blur, threshold_1, threshold_2)
    # Show image
    plt.imshow(canny, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    plt.show()
    # Save image into the assignment_3 folder
    cv2.imwrite(str(path) + '/lambo_canny_SvenjaJetzinger_A3.png', canny)


# Template match
def template_match(picture, template, path):
    # Convert both images to grayscale
    picture_gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # Get the width and height of the template
    h, w = template_gray.shape
    # Perform template matching
    result = cv2.matchTemplate(picture_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    # Set a threshold for detecting matches
    threshold = 0.9
    loc = np.where(result >= threshold)
    # Draw red rectangles around detected matches -> found all left corners of rectangles because of grayscale; not possible in RGB!
    for pt in zip(*loc[::-1]):                                                  # Switch columns and rows
        cv2.rectangle(picture, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)      # Rot: (b=0, g=0, r=255)
    # Show image
    plt.imshow(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
    plt.title('Template Matching')
    plt.axis('off')
    plt.show()
    # Save image into the assignment_3 folder
    cv2.imwrite(str(path) + '/template_match_SvenjaJetzinger_A3.png', picture)
    



main()