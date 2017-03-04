import cv2
import numpy as np


def main():
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	cam = cv2.VideoCapture(-1)
	while(True):
		ret, frame = cam.read()
		grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		img = grayscale

		faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = grayscale[y:y+h, x:x+w]
    			roi_color = img[y:y+h, x:x+w]
    			eyes = eye_cascade.detectMultiScale(roi_gray)
    			for (ex,ey,ew,eh) in eyes:
    				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

		cv2.imshow("camera", img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
        		break

	cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()