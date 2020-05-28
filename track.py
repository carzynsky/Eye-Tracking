import cv2
import numpy as np
import csv
import pandas as pd

# Klasyfikatory
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# parametry detektora źrenic
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector_params.filterByConvexity = False
detector_params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(detector_params)

# xy lewego gornego rogu wykrytej twarzy
detected_face_cord = [0,0]

# progowanie
left_eye_best_th = 30
right_eye_best_th = 30

# kierunek strojenia
directionL = 1
directionR = 1

# wspolrzedne rogów oczu
leftEyeCornerLeft = [0,0]
leftEyeCornerRight = [0,0]
rightEyeCornerLeft = [0,0]
rightEyeCornerRight = [0,0]

# środek oczu na podstawie rogów
centreOfLeftEye = [0,0]
centreOfRightEye = [0,0]

# wymiary okna
testWidth = 0
testHeight = 0

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
        detected_face_cord[0] = x
        detected_face_cord[1] = y
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

    # xy wykrytych źrenic / środek masy tęczówek
    detected_left_blob_cord = [0,0]
    detected_right_blob_cord = [0,0]
    RightR = [0,0]
    LeftR = [0,0]   
    
    global rightEyeCornerRight
    global rightEyeCornerLeft
    global leftEyeCornerRight
    global leftEyeCornerLeft
    global centreOfRightEye
    global centreOfRightEye

    eyes = eye_cascade.detectMultiScale(face_gray, 1.30, 8)

    for (x, y, w, h) in eyes:
        # Pominięcie 'wykrytych blednie oczy' poniżej połowy wysokosci twarzy
        if  y <=  (height*0.5):
            face_color = cv2.rectangle(face_color, (x, y), (x + w, y + h), (0,255,0), 2)    
            eye_center = (x + w/2)
            # Lewe oko
            if eye_center > (width/2):
                global left_eye_best_th
                left_eye = face_color[y:y+h, x:x+w]
                left_eye = cut_eye(left_eye)
                l_eye_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
                height_e, width_e = left_eye.shape[:2]

                # left corner
                l_eye_gray = l_eye_gray[2*int(height_e/5):4*int(height_e/5), 0:int(width_e/4)]
                l_eye = left_eye[2*int(height_e/5):4*int(height_e/5), 0:int(width_e/4)]
                corners = cv2.goodFeaturesToTrack(l_eye_gray, 1, 0.1, 20)

                if corners is not None:
                    corners = np.int0(corners)
                    for corner in corners:
                        xc, yc = corner.ravel()
                        leftEyeCornerLeft[0] = detected_face_cord[0] + x + xc
                        leftEyeCornerLeft[1] = detected_face_cord[1] + y + yc + 2*int(height_e/5)
                        cv2.circle(l_eye, (xc, yc), 2, (255,255,0), -1)

                #cv2.imshow('left_eye',l_eye)  

                # right corner
                left_eye_right_corner_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
                left_eye_right_corner_gray = left_eye_right_corner_gray[int(height_e/2):height_e, 3*int(width_e/4):width_e]
                l2_eye = left_eye[int(height_e/2):height_e, 3*int(width_e/4):width_e]               
                corners2 = cv2.goodFeaturesToTrack(left_eye_right_corner_gray, 1, 0.1, 20)
                if corners2 is not None:
                    corners2 = np.int0(corners2)              
                    for corner in corners2:
                        xc, yc = corner.ravel()
                        leftEyeCornerRight[0] = detected_face_cord[0] + x + xc + 3*int(width_e/4)
                        leftEyeCornerRight[1] = detected_face_cord[1] + y + yc + int(height_e/2)
                        cv2.circle(l2_eye, (xc, yc), 2, (255,255,0), -1)

                keypoints = blob_process(left_eye, left_eye_best_th, detector) # th = 30

                if(len(keypoints) == 0):
                    global directionL
                    if(left_eye_best_th == 60 or left_eye_best_th == 20):
                        directionL = directionL*(-1)
                    left_eye_best_th += 1*directionL

                if(len(keypoints) > 1):
                    if(directionL == 1):
                        directionL = -1
                    left_eye_best_th += 1*directionL
                    print('zmieniam na {}'.format(left_eye_best_th))
                
                if(len(keypoints) == 1):
                    detected_left_blob_cord[0] = detected_face_cord[0] + x + keypoints[0].pt[0]
                    detected_left_blob_cord[1] = detected_face_cord[1] + y + keypoints[0].pt[1]

                centreOfLeftEye[0] = (leftEyeCornerLeft[0] + leftEyeCornerRight[0])/2
                centreOfLeftEye[1] = (leftEyeCornerLeft[1] + leftEyeCornerRight[1])/2
                key_reset_left_th = cv2.waitKey(30)
                if key_reset_left_th == 51:
                    left_eye_best_th = 25
                    directionL = 1

                left_eye = cv2.drawKeypoints(left_eye, keypoints, left_eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.putText(face_color, 'Left eye', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

            # Prawe oko
            else:
                global right_eye_best_th
                right_eye = face_color[y:y+h, x:x+w]
                right_eye = cut_eye(right_eye)
                # right corner
                r_eye_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
                height_e, width_e = right_eye.shape[:2]
                r_eye_gray = r_eye_gray[2*int(height_e/5):4*int(height_e/5), 3*int(width_e/4):width_e]
                r_eye = right_eye[2*int(height_e/5):4*int(height_e/5), 3*int(width_e/4):width_e]
                cv2.imshow('right_eye',r_eye)

                corners = cv2.goodFeaturesToTrack(r_eye_gray, 1, 0.1, 20)
                if corners is not None:
                    corners = np.int0(corners)
                    for corner in corners:
                        xc, yc = corner.ravel()
                        rightEyeCornerRight[0] = detected_face_cord[0] + x + xc + 3*int(width_e/4)
                        rightEyeCornerRight[1] = detected_face_cord[1] + y + yc + 2*int(height_e/5)
                        cv2.circle(r_eye, (xc, yc), 2, (255,255,0), -1)
                #cv2.imshow('right_eye', r_eye)
                # left corner
                r2_eye_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
                r2_eye_gray = r2_eye_gray[2*int(height_e/5):4*int(height_e/5), 0:int(width_e/3)]
                r2_eye = right_eye[2*int(height_e/5):4*int(height_e/5), 0:int(width_e/3)]
                #cv2.imshow('right_eye', r2_eye)
                corners2 = cv2.goodFeaturesToTrack(r2_eye_gray, 1, 0.1, 20)
                if corners2 is not None:
                    corners2 = np.int0(corners2)
                    for corner in corners2:
                        xc, yc = corner.ravel()
                        rightEyeCornerLeft[0] = detected_face_cord[0] + x + xc
                        rightEyeCornerLeft[1] = detected_face_cord[1] + y + yc + 2*int(height_e/5)
                        cv2.circle(r2_eye, (xc, yc), 2, (255,255,0), -1)

                
                centreOfRightEye[0] = (rightEyeCornerLeft[0]+rightEyeCornerRight[0])/2
                centreOfRightEye[1] = (rightEyeCornerLeft[1] + rightEyeCornerRight[1])/2
               # print("ceo right x = {}, y = {}".format(centreOfRightEye[0], centreOfRightEye[1]))

                keypoints = blob_process(right_eye, right_eye_best_th, detector) # th=30

                if(len(keypoints) == 0):
                    global directionR
                    if(right_eye_best_th == 60 or right_eye_best_th == 20):
                        directionR = directionR * (-1)
                    right_eye_best_th += 1*directionR
                    #print('new right th: {}'.format(right_eye_best_th))

                if(len(keypoints) > 1):
                    if(directionR == 1):
                        directionR = -1
                    right_eye_best_th += 1*directionR

                if(len(keypoints) == 1):
                    detected_right_blob_cord[0] = detected_face_cord[0] + x + keypoints[0].pt[0]
                    detected_right_blob_cord[1] = detected_face_cord[1] + y + keypoints[0].pt[1]
                    #print('right blob x={}, y={}'.format(detected_right_blob_cord[0], detected_right_blob_cord[1]))

                key_reset_right_th = cv2.waitKey(30)
                if key_reset_right_th == 51:
                    right_eye_best_th = 25
                    directionR = 1

                #print('kp = {}'.format(cv2.KeyPoint_convert(keypoints)))
                right_eye = cv2.drawKeypoints(right_eye, keypoints, right_eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.putText(face_color, 'Right eye', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    # obliczenie punktu skupienia wzroku
    leftEyeWidth = leftEyeCornerRight[0] - leftEyeCornerLeft[0] # szerokość lewego oka
    leftEyeHeight = leftEyeWidth * 0.28 # wysokosc oka stanowi 28/32 % szerokosci oka
    rightEyeWidth = rightEyeCornerRight[0] - rightEyeCornerLeft[0]
    rightEyeHeight = rightEyeWidth * 0.28 
    avgWidth = (leftEyeWidth + rightEyeWidth)/2 # średnia szerokość oka
    avgHeight = (leftEyeHeight + rightEyeHeight)/2 # średnia wysokosc oka
    RxFactor = testWidth/avgWidth # skala
    RyFactor = testHeight/avgHeight
    RightR[0] =  detected_right_blob_cord[0] - centreOfRightEye[0] # ????????
    RightR[1] =  detected_right_blob_cord[1] - centreOfRightEye[1]
    LeftR[0] =   detected_left_blob_cord[0] - centreOfLeftEye[0]
    LeftR[1] = detected_left_blob_cord[1] - centreOfLeftEye[1]
    RxAvg = (RightR[0] + LeftR[0])/2
    RyAvg = (RightR[1] + LeftR[1])/2
    ProjectionX = int((testWidth/2) + RxFactor * RxAvg)
    ProjectionY = int((testHeight/2) + RyFactor * RyAvg)
    return ProjectionX, ProjectionY

def cut_eye(eye):
    height, width = eye.shape[:2]
    eyebrow_y = int(height/5)
    eye = eye[eyebrow_y:height-eyebrow_y, 0:width]
    return eye

def blob_process(eye, th, detector):
    gray_roi = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7,7), 0)
    _, threshold =  cv2.threshold(gray_roi, th, 255, cv2.THRESH_BINARY)
    img = cv2.erode(threshold, None, iterations=2) #1
    img = cv2.dilate(img, None, iterations=4) #2
    img = cv2.medianBlur(threshold, 5) #3
    keypoints = detector.detect(img)
    return keypoints

def nothing(x):
    pass

def main():
    # video = cv2.VideoCapture(0)
    # video.open(0);

    video = cv2.VideoCapture('videos/virtual.mp4') # for video from file, change parameter to camera port
    global testWidth
    global testHeight

    testWidth  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    testHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pList = []  # lista wspolrzednych punktu skupienia wzroku

    cv2.namedWindow('image')
    print('width = {}, height = {}'. format(testWidth, testHeight))
    # cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    start_detect_face = True
    while True:
        ret, frame = video.read()
        if ret is False:
            break
        
        key_detect_face = cv2.waitKey(30)
        
        if key_detect_face == 49:
            start_detect_face = True

        if key_detect_face == 50:
            start_detect_face = False

        if(start_detect_face):       
            face_frame = face_detection(frame, face_cascade)
            if face_frame is not None:
                ProjectionX, ProjectionY = detect_eyes(face_frame, eye_cascade)
                pList.append([ProjectionX,ProjectionY])
                cv2.circle(frame, (ProjectionX, ProjectionY), 5, (255,0,255), -1)      
        
        # Top Left text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText = (10,20)
        fontScale = 0.4
        fontColor = (0,0,255)
        lineType = 2
        cv2.putText(frame,'Click 1 to detect, 2 to stop detecting, 3 to reset threshold, ESC to exit', topLeftCornerOfText, font, fontScale,fontColor,lineType)
        cv2.imshow('image', frame) 
        
        key = cv2.waitKey(30)
        if key == 27: 
            with open("data.csv", "w", newline='') as csv_file:   
                writer = csv.writer(csv_file, delimiter=',')
                for x in pList:
                    writer.writerow((x[0], x[1]))
            break;
    cv2.destroyAllWindows()
main()
