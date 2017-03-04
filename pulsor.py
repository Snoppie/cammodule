import cv2
import numpy as np


def get_subface_coord(fh_x, fh_y, fh_w, fh_h, face_rect):
    x, y, w, h = face_rect
    return [int(x + w * fh_x - (w * fh_w / 2.0)),
                    int(y + h * fh_y - (h * fh_h / 2.0)),
                    int(w * fh_w),
                    int(h * fh_h)]


def get_subface_means(coord, frame):
    x, y, w, h = coord
    subframe = frame[y:y + h, x:x + w, :]
    v1 = np.mean(subframe[:, :, 0])
    v2 = np.mean(subframe[:, :, 1])
    v3 = np.mean(subframe[:, :, 2])

    return (v1 + v2 + v3) / 3.


def shift(detected):
    x, y, w, h = detected
    center = np.array([x + 0.5 * w, y + 0.5 * h])
    newshift = np.linalg.norm(center - shift.last_center)

    shift.last_center = center
    return newshift


shift.last_center = np.array([0,0])


def main():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    cam = cv2.VideoCapture(-1)
    face_rect = [1, 1, 2, 2]
    while(True):
        ret, frame = cam.read()
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = frame
        faces = list(face_cascade.detectMultiScale(grayscale, 1.3, 5))
                

        # Only update rect if face has shifted
        if len(faces) > 0:

                        # Sort faces
                        faces.sort(key=lambda a: a[-1] * a[-2])

                        if shift(faces[-1]) > 10:
                            face_rect = faces[-1]

        (x,y,w,h) = face_rect
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                
        # Draw forehead region
        forehead = fx, fy, fw, fh = get_subface_coord(0.5, 0.18, 0.25, 0.15, (x, y, w, h))
        cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (0,0,255), 2)

        roi_gray = grayscale[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # Draw all eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        # Here goes code to calculate the pulse
        vals = get_subface_means(forehead, frame)
        cv2.putText(img, str(vals), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (100, 250, 100))

        cv2.imshow("camera", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
