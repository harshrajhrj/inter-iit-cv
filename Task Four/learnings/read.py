import cv2 as cv

# img = cv.imread('photos/01.png')

# Reading images
# window name, image var name 
# cv.imshow('photo', img)

# keyboard binding function, adds a delay until window is closed
# cv.waitKey(0)

# Reading videos
# VideoCapture args: 0, 1, 2, 3, path => 0 is webcam, 1 is first cam, 2 is second cam and so on
# ...................we can include path
# capture is an instance of VideoCapture class
capture = cv.VideoCapture('videos/dog.mp4')

while True:
    isTrue, frame = capture.read() # returns frame and boolean (True - successfully read/False - failed)
    cv.imshow('video', frame)
    # we might get an error if the video goes out of frames or capture couldn't find the media file
    # video stops when letter 'D' is pressed
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()