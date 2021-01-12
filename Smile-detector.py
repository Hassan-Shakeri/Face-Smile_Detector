import cv2

#load some pre-trained data on face frontals from opencv 
#load some pre-trained data on smile from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_smile_data = cv2.CascadeClassifier('haarcascade_smile.xml')

#cpture video from webcame
webcam = cv2.VideoCapture(0)

#interact forever over frames
while True:

	#read the current frame
	successful_frame_read, frame = webcam.read()

	#if there is an error abort
	if not successful_frame_read:
		break

	#convert to grayscale
	grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#detect faces
	face_cordinates = trained_face_data.detectMultiScale(grayscaled_img)
	
	#run face detection within each face
	for (x, y, w, h) in face_cordinates:

		#draw rectangles around the face
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)

		#get the sub frame (using numpy N-dimentional array slicing)
		the_face = frame[y:y+h, x:x+w]

		#change to grayscale
		face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

		smiles = trained_smile_data.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

		#find all the smiles in the face
		#for (x_,y_,w_,h_) in smiles:
			#draw a rectangle around the smile
			#cv2.rectangle(the_face, (x_, y_), (x_ + w_,y_ + h_), (0,0,255), 3)

		#label the face as smiling
		if len(smiles) > 0:
			cv2.putText(frame, 'Smiling...', (x,y+h+40), fontScale= 3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))

	#show the current frame
	cv2.imshow("Hassan's face-detector app", frame)
	key = cv2.waitKey(1)

	#stop if Q is pressed (using asciii number)
	if key==81 or key==113:
		break

#release the video capture object
webcam.release()
cv2.destroyAllWindows


print("Code Completed")