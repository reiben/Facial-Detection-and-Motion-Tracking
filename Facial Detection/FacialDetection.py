import cv2

cap = cv2.VideoCapture(0) # Class which helps to capture frame by frame
face_classifier = cv2.CascadeClassifier('haarcascade_frontalcatface.xml') # Data for trained face detection model
profile_face_classifier = cv2.CascadeClassifier('haarcascade_profileface.xml')

if cap.isOpened() == False: # if the VideoCapture unables to start video an error is displayed
	print("An error occured")

while cap.isOpened():
	ret,frame = cap.read() # Reads the Video Frame by frame
	faces = face_classifier.detectMultiScale(frame,scaleFactor=1.2,minNeighbors=6) # This functions helps to detect multiple faces from classifier
	profiles = profile_face_classifier.detectMultiScale(frame,scaleFactor=1.2,minNeighbors=6) # Detects face profiles
	for (x1,y1,w1,h1) in faces:
		cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0)) # Creates a rectangle where the face coordinates are present
	for (x2,y2,w2,h2) in profiles:
		cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(255,0,0))
	cv2.imshow("Video",frame) # Shows the video at each frame
	key = cv2.waitKey()
	if key == ord('q'):
		break
cap.release() # Releases the Video
cv2.destroyAllWindows()
