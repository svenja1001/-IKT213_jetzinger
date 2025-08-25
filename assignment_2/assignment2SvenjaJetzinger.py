import numpy as np
import cv2
from matplotlib import pyplot as plt

def main():
    path = 'assignment_#. Ex. nnoori/IKT213_noori/assignment_2'
    img = cv2.imread(str(path) + '/lena-2.png')                     # read/load the image
    
    # Border width for padding
    border_width = 100
    padding(img, border_width)

    # Coordinates for cropping
    x_0 = 80                                                        # space from left side of the picture    
    x_1 = img.shape[1] - 130                                        # space from right side of the picture (picture width - space)
    y_0 = 80                                                        # space from top side of the picture
    y_1 = img.shape[0] - 130                                        # space from bottom side of the picture (picture height - space)                        
    crop(img, x_0, x_1, y_0, y_1)

    # Pixels for resizing
    resize_width = 200
    resize_height = 200
    resize(img, resize_width, resize_height)

    # Create an empty picture array for manual copy
    copy_height = img.shape[0]                                      # get the height of the original image
    copy_width = img.shape[1]                                       # get the width of the original image
    emptyPictureArray = np.zeros((copy_height, copy_width, 3), dtype=np.uint8)  # create an empty picture array with the same size as the original image
    copy(img, emptyPictureArray)
    
    grayscale(img)
    
    hsv(img)

    # Color shifting
    hue = 50
    hue_shifted(img, emptyPictureArray, hue)

    smoothing(img)

    rotation_angle = 180                                           # rotation angle in degrees (90, 180, or any other angle)
    rotation(img, rotation_angle)
    
    plt.show()                                                      # show all images in one window


# Padding
def padding(img, border_width):
    reflection = cv2.copyMakeBorder(img, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)    
    reflection_rgb = cv2.cvtColor(reflection, cv2.COLOR_BGR2RGB)    # convert to RGB -> OpenCV uses BGR, matplotlib RGB
    plt.subplot(3, 3, 1)
    plt.imshow(reflection_rgb)
    plt.title('Reflection Padding')
    plt.axis('off')


# Cropping
def crop(img, x_0, x_1, y_0, y_1):
    cropped = img[y_0:y_1, x_0:x_1]                                 # crop the image
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)          # convert to RGB -> OpenCV uses BGR, matplotlib RGB
    plt.subplot(3, 3, 2)
    plt.imshow(cropped_rgb) 
    plt.title('Cropped Image')
    plt.axis('off')
    

# Resizing
def resize(img, resize_width, resize_height):
    resized = cv2.resize(img, (resize_width, resize_height))        # resize the image
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)          # convert to RGB -> OpenCV uses BGR, matplotlib RGB
    plt.subplot(3, 3, 3)
    plt.imshow(resized_rgb)
    plt.title('Resized Image')
    plt.axis('off')


# Manual copy
def copy(img, emptyPictureArray):
    emptyPictureArray[:] = img[:]                                   # Copies all pixel values from the original image to the empty array   
    copied_rgb = cv2.cvtColor(emptyPictureArray, cv2.COLOR_BGR2RGB) # convert to RGB -> OpenCV uses BGR, matplotlib RGB
    plt.subplot(3, 3, 4)
    plt.imshow(copied_rgb)
    plt.title('Manual Copy')        
    plt.axis('off')


# Grayscale
def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                    # convert the image from RGB to grayscale
    plt.subplot(3, 3, 5)
    plt.imshow(gray, cmap='gray')                                   # display the grayscale image
    plt.title('Grayscale Image')
    plt.axis('off')


# HSV -> Hue (Farbton), Saturation (Sättigung), Value (Helligkeit)
def hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                  # convert the image from RGB to HSV
    plt.subplot(3, 3, 6)
    plt.imshow(hsv_img)
    plt.title('HSV Image')
    plt.axis('off')


# Color shifting
def hue_shifted(img, emptyPictureArray, hue):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                  # change picture in HSV
    hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] + hue, 0, 255)      # shift value-channel and limit values to [0, 255] 
    shifted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)          # convert back to BGR
    emptyPictureArray[:] = shifted_img[:]                           # copy result to empty array
    shifted_rgb = cv2.cvtColor(shifted_img, cv2.COLOR_BGR2RGB)      # convert for matplotlib
    plt.subplot(3, 3, 7)
    plt.imshow(shifted_rgb)
    plt.title(f'Value Shifted ({hue})')
    plt.axis('off')


# Smoothing
def smoothing(img):
    smoothed = cv2.GaussianBlur(img, (15, 15), 0)                   # apply Gaussian blur with kernel size 15x15
    smoothed_rgb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB)        # convert for matplotlib
    plt.subplot(3, 3, 8)
    plt.imshow(smoothed_rgb)
    plt.title('Smoothed Image')
    plt.axis('off')


# Rotation
def rotation(img, rotation_angle):
    if rotation_angle == 90:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(img, cv2.ROTATE_180)
    else:
        center = (img.shape[1] // 2, img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
    rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 3, 9)
    plt.imshow(rotated_rgb)
    plt.title(f'Rotated Image ({rotation_angle}°)')
    plt.axis('off')
    

main()
