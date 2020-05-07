import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector_params.filterByConvexity = False
detector_params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(detector_params)


# faces detection
def face_detection(frame, classifier):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray_frame, 1.30, 5)

    biggest = (0,0,0,0)
    frame_face = None

    if len(faces) > 1:
        for x in faces:
            if(x[3] > biggest[3]):
                biggest = x
        biggest = np.array([x], np.int32)
    elif (len(faces) == 1):
        biggest = faces
    else:
        return None

    for (x,y,w,h) in biggest:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        frame_face = frame[y:y + h, x:x + w]
    return frame_face


# eyes detection
def detect_eyes(face_color, eye_cascade):

    face_gray = cv2.cvtColor(face_color, cv2.COLOR_BGR2GRAY)
    width = np.size(face_color, 1)
    height = np.size(face_color, 0)

    left_eye = None
    right_eye = None

    eyes = eye_cascade.detectMultiScale(face_gray, 1.30, 8)

    for (x, y, w, h) in eyes:
        # Pominięcie 'wykrytych blednie oczy' poniżej połowy wysokosci twarzy
        if  y <=  (height*0.5):
            face_color = cv2.rectangle(face_color, (x, y), (x + w, y + h), (0,255,0), 2)    
            eye_center = (x + w/2)

            if eye_center > (width/2):
                left_eye = face_color[y:y+h, x:x+w]
                left_eye = cut_eye(left_eye)
                keypoints = blob_process(left_eye, 30, detector)
                left_eye = cv2.drawKeypoints(left_eye, keypoints, left_eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.putText(face_color, 'Left eye', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            else:
                right_eye = face_color[y:y+h, x:x+w]
                right_eye = cut_eye(right_eye)
                keypoints = blob_process(right_eye, 30, detector)
                right_eye = cv2.drawKeypoints(right_eye, keypoints, right_eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.putText(face_color, 'Right eye', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
         
    return left_eye, right_eye


def cut_eye(eye):
    height, width = eye.shape[:2]
    # new_x = int(width/(4))
    eyebrow_y = int(height/4)
    eye = eye[eyebrow_y:height-eyebrow_y, 0:width]

    return eye


def blob_process(eye, th, detector):
    gray_roi = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7,7), 0)

    _, threshold =  cv2.threshold(gray_roi, th, 255, cv2.THRESH_BINARY)

    img = cv2.erode(threshold, None, iterations=2) #1
    img = cv2.dilate(img, None, iterations=4) #2
    img = cv2.medianBlur(threshold, 5) #3
    cv2.imshow('img',img)
    keypoints = detector.detect(img)
    return keypoints

def nothing(x):
    pass


def main():
    video = cv2.VideoCapture('videos/shake.mp4') # for video from file, change parameter to camera port
    # video.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cv2.namedWindow('image')
    # cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

    while True:
        ret, frame = video.read()
        if ret is False:
            break
        
        face_frame = face_detection(frame, face_cascade)

        if face_frame is not None:
            detect_eyes(face_frame, eye_cascade)
            # for eye in eyes:
            #     if eye is not None:
            #         # threshold = cv2.getTrackbarPos('threshold', 'image')
            #         eye = cut_eyebrows(eye)
            #         keypoints = blob_process(eye, 65, detector)
            #         eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    # for contour in contours:
                    #     cv2.drawContours(eye, [contour], -1, (0,0,255), 3)
            
        cv2.imshow('image', frame) 
        
        key = cv2.waitKey(30)
        if key == 27: 
            break;
    cv2.destroyAllWindows()

main()
