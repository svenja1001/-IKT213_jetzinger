import numpy as np
import cv2

def main():
    path = 'assignment_1'
    img = cv2.imread(str(path) + '/lena-1.png')             # read an image
    print_image_information(img)

    video()
    
    

def print_image_information(img):

    print('height: ' + str(img.shape[0]))                   # print the height of the image
    print('width: ' + str(img.shape[1]))                    # print the width of the image
    print('channels: ' + str(img.shape[2]))                 # print the number of channels in the image
    print('size: ' + str(img.size))                         # print the total number of values in the array
    print('data type: ' + str(img.dtype))                   # print the data type of the image array

def video():
    
    cam = cv2.VideoCapture(0)                               # Open the default camera
    
    # frame per seconds
    fps = cam.get(cv2.CAP_PROP_FPS)                         # Get the frames per second of the camera       
    print('FPS: ' + str(fps))                               # Print the frames per second (fps)

    # frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('Frame Width: ' + str(frame_width))               # Print the width of the frame
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Frame Height: ' + str(frame_height))             # Print the height of the frame

    cam.release()                                           # Release (close) the camera

main()

