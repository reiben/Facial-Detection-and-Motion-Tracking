import cv2
import numpy as np


cap = cv2.VideoCapture("vtest.avi") # Class which helps to capture frame by frame
_,frame1 = cap.read() # Reads a paricular frame
_,frame2 = cap.read()

if cap.isOpened() == False: # if the VideoCapture unables to start video an error is displayed
    print("Error Displaying video")

while cap.isOpened():
    diff = cv2.absdiff(frame1,frame2) # Finds the difference in the frame to detect motion
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY) # Converts to grayscale for blurring and dilating
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _,th = cv2.threshold(blur,20,255,cv2.THRESH_BINARY) # Binary Threshold on the blurred grayscale frame
    dialate = cv2.dilate(th,None,iterations=3) # Dialates the image upto 3 iterations
    countours,_ = cv2.findContours(dialate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in countours:
        (x,y,w,h) = cv2.boundingRect(i) # Returns the poins, height, Width for creating rectangle
        if cv2.contourArea(i) < 700: # If the area where motion is detected is less than 700 then that part is ignored
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),4)
    #ret,frame = cap.read()
    cv2.imshow("Video",frame1)
    frame1 = frame2
    _,frame2 = cap.read()
    key = cv2.waitKey()
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()