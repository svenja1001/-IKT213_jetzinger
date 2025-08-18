import numpy as np
import cv2

def main():
    path = 'assignment_#. Ex. nnoori/IKT213_noori/assignment_1'
    img = cv2.imread(str(path) + '/lena-1.png')             # read an image
    print_image_information(img)
    

def print_image_information(img):

    print('height: ' + str(img.shape[0]))                   # print the height of the image
    print('width: ' + str(img.shape[1]))                    # print the width of the image
    print('channels: ' + str(img.shape[2]))                 # print the number of channels in the image
    print('size: ' + str(img.size))                         # print the total number of values in the array
    print('data type: ' + str(img.dtype))                   # print the data type of the image array

main()

