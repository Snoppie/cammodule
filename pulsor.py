import cv2
import time
import numpy as np
from requests import Session
import json


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


def shift(detected, old_detected):
  x, y, w, h = detected
  center = np.array([x + 0.5 * w, y + 0.5 * h])
  x, y, w, h = old_detected
  last_center = np.array([x + 0.5 * w, y + 0.5 * h])
  shift = np.linalg.norm(center - last_center)

  return shift


def main():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    #cam = cv2.VideoCapture(-1)
    cam = cv2.VideoCapture("http://localhost:1234")
    face_rect = [1, 1, 2, 2]
    times = list()
    data_buffer = list() 
    t0 = time.time()
    # loops since last refresh
    refreshc = 0
    historic_bpm = 0
    pulse_history = list()
    json_output = {"id": 0, "data":{}}
    session = Session()
    while(True):
        times.append(time.time() - t0)
        ret, frame = cam.read()
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = frame
        faces = list(face_cascade.detectMultiScale(grayscale, 1.3, 5))
                

        # Only update rect if face has shifted
        if len(faces) > 0:

                        # Sort faces
                        faces.sort(key=lambda a: a[-1] * a[-2])


                        if shift(faces[-1], face_rect) > 10:
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
        data_buffer.append(vals)
        L = len(data_buffer)
        if L > 250:
            data_buffer = data_buffer[-250:]
            times = times[-250:]
            L = 250

        processed = np.array(data_buffer)
        if L > 10:
            fps = float(L) / (times[-1] - times[0])
            even_times = np.linspace(times[0], times[-1], L)
            interpolated = np.interp(even_times, times, processed) ##here
            interpolated = np.hamming(L) * interpolated ##here
            interpolated = interpolated - np.mean(interpolated) ##here
            raw = np.fft.rfft(interpolated) ##here
            fft = np.abs(raw) ##here
            freqs = float(fps) / L * np.arange(L / 2 + 1) ##here
            freqs = 60. * freqs ##here
            idx = np.where((freqs > 50) & (freqs < 180)) ##here
            pruned = fft[idx] ##here

            pfreq = freqs[idx] ##here
            freqs = pfreq ##here
            fft = pruned   ##here
            idx2 = np.argmax(pruned)  ##here
            bpm = freqs[idx2]


            refreshc += 1
            pulse_history.append(bpm)


            if refreshc >= 10:
                bpm = np.mean(pulse_history)
                historic_bpm = bpm
                pulse_history = list()
                refreshc = 0
                json_output["data"]["bpm"] = bpm
                json_output["data"]["face"] = len(faces) > 0
                session.post("http://ec2-54-93-71-88.eu-central-1.compute.amazonaws.com/update", data=json.dumps(json_output))
                    
            else:
                bpm = historic_bpm

            cv2.putText(img, "fps: "+ str(fps), (50, 360), cv2.FONT_HERSHEY_PLAIN, 1, (100, 250, 100))
            cv2.putText(img, "interp: "+ str(interpolated), (50, 370), cv2.FONT_HERSHEY_PLAIN, 1, (100, 250, 100))
            cv2.putText(img, "raw: "+ str(raw), (50, 380), cv2.FONT_HERSHEY_PLAIN, 1, (100, 250, 100))
            cv2.putText(img, "fft: "+ str(fft), (50, 390), cv2.FONT_HERSHEY_PLAIN, 1, (100, 250, 100))
            cv2.putText(img, "freqs: "+ str(freqs), (50, 400), cv2.FONT_HERSHEY_PLAIN, 1, (100, 250, 100))
            cv2.putText(img, "idx: "+ str(idx), (50, 410), cv2.FONT_HERSHEY_PLAIN, 1, (100, 250, 100))
            cv2.putText(img, "pruned: "+ str(pruned), (50, 420), cv2.FONT_HERSHEY_PLAIN, 1, (100, 250, 100))
            cv2.putText(img, "idx2: "+ str(idx2), (50, 430), cv2.FONT_HERSHEY_PLAIN, 1, (100, 250, 100))
            cv2.rectangle(img, (50,420), (500,440),(230,230,230), -1)
            cv2.putText(img, "** -> PULS: "+ str(np.round(bpm, 2)), (50, 440), cv2.FONT_HERSHEY_PLAIN, 1.5, (10, 10, 250))


        #cv2.imshow("camera", img)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(buf.tostring())

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
