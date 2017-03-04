import cv2
import numpy as np


def main():
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	cam = cv2.VideoCapture(-1)
	while(True):
		ret, frame = cam.read()
		grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		img = grayscale

		faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)
		for (x,y,w,h) in faces:
		    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

		cv2.imshow("camera", img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
        		break

	cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()