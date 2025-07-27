# When files are large it is necessary to resize and rescale files to make processing faster although some information are lost
# Resizing means modifying the height and width to a particular height and width
# Best practice to downscale height and width of a video file

import cv2 as cv

# Reading image
img = cv.imread('photos/01.png')
cv.imshow('photo', img)

def changeRes(width, height):
    """
    Only for live videos
    """
    # 3 and 4 are properties of capture class
    capture.set(3, width) # 3: width
    capture.set(4, height) # 4: height

def rescale_img(frame, scale=0.75):
    """
    Specifically for videos (also images and live video)
    """
    width = int(frame.shape[1] * scale) # .shape[1] = width
    height = int(frame.shape[0] * scale) # .shape[0] = height
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# resized_image = rescale_img(img)
# cv.imshow('photo_resized', resized_image)

# Reading videos
capture = cv.VideoCapture('video/dog.mp4')

while True:
    isTrue, frame = capture.read() # returns frame and boolean (True - successfully read/False - failed)
    
    frame_resize = rescale_img(frame)

    cv.imshow('video', frame)
    cv.imshow('video_resize', frame_resize)

    # we might get an error if the video goes out of frames or capture couldn't find the media file
    # stops when letter 'D' is pressed
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()