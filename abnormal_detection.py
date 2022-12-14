
# coding: utf-8
# In[1]:

import numpy as np
import imutils
import time
import timeit
import dlib
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from threading import Timer
from check_cam_fps import check_fps
import make_train_data as mtd
import light_remover as lr
import ringing_alarm as alarm
import serial

'''
얼굴와 안구 landmark 검출 기법(HOG)
1) Dlib의 HOG
2) 68 face landmark estimation

눈 감음 정도 검출 기법
1) EAR(Eye Aspect Ratio; 눈 영상비) 기법
- 2016년 Tereza Soukupova & Jan'Cech에 의해 제시된 EAR 방식
- 사용자의 눈 개폐상태를 확인(눈을 감으면 EAR 비율이 감소)
- 평소 눈의 Threshold(THRESH);임계값을 설정한 후 이를 기준으로 기수면상태 확인
'''
#####################################################################################################################

MT = serial.Serial('/dev/ttyACM1', 9600)  # Motor, LED 제어 Arduino 시리얼 포트 설정
BT = serial.Serial('/dev/ttyACM0', 9600)  # Bluetooth 통신 Arduino 시리얼 포트 설정

# <함수 정의>
# (초기 EAR 설정부)
#1. eye_aspect_ratio: 68 face landmark 중 6개의 eye landmark 기반으로 EAR 비율 도출
#2. init_open_ear: EAR_THRESH 도출을 위한 EAR 최대 평균값(OPEN_EAR) 계산
#3. init_close_ear: EAR_THRESH 도출을 위한 EAR 최소 평균값(CLOSE_EAR) 계산 및 EAR_THRESH 도출(50%)
# (제어부)
#4.
#5.
#6.
#7.

#1. 68 face landmark 중 6개의 eye landmark 기반으로 EAR 비율 도출
def eye_aspect_ratio(eye) :
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

#2. EAR_THRESH 도출을 위한 EAR 최대 평균값(OPEN_EAR) 계산
def init_open_ear() :
    time.sleep(5)   # sleep 함수 = 일시정지(time.sleep(5)=5초 일시정지)
    print("Measuring init OPEN EAR")
    ear_list = [] # init_open_ear 평균 도출을 위한 EAR값 표본 리스트
    th_message1 = Thread(target = init_message_open)
    th_message1.deamon = True
    th_message1.start()
    # 과정 1: 운전자가 눈을 뜨고 있을 때의 평균 EAR 값 결정(OPEN_EAR)
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    global OPEN_EAR
    OPEN_EAR = sum(ear_list) / len(ear_list) # 측정한 EAR 값의 평균 값이 저장될 변수
    print("open list =", ear_list, "\nOPEN_EAR =", OPEN_EAR, "\n")
    alarm.sound_alarm("finish_init.wav")

#3. EAR_THRESH 도출을 위한 EAR 최소 평균값(CLOSE_EAR) 계산 및 EAR_THRESH 도출(50%)
def init_close_ear() : 
    time.sleep(2)
    th_open.join()
    time.sleep(5)
    print("Measuring init CLOSE EAR")
    ear_list = []
    th_message2 = Thread(target = init_message_close)
    th_message2.deamon = True
    th_message2.start()
    time.sleep(1)
    # 과정 2: 운전자가 눈을 감고 있을 때 평균 EAR 값 결정(CLOSE_EAR)
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    CLOSE_EAR = sum(ear_list) / len(ear_list)
    # 과정 3: 과정 1, 2의 결과 값(OPEN_EAR/CLOSE_EAR) 이용해 EAR의 50% 값을 EAR_THRESH로 설정
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR) #EAR_THRESH means 50% of the being opened eyes state
    print("close list =", ear_list, "\nCLOSE_EAR =", CLOSE_EAR, "\n")
    print("The EAR_THRESH value :",EAR_THRESH, "\n")
    alarm.sound_alarm("finish_init.wav")
    time.sleep(3)
    alarm.sound_alarm("start_process.wav")

#4. 초기 OPEN_EAR 값 설정 음성안내
def init_message_open() :
    print("init_message_OPEN")
    alarm.sound_alarm("init_sound_open.wav")

#5. 초기 CLOSE_EAR 값 설정 음성안내
def init_message_close() :
    print("init_message_CLOSE")
    alarm.sound_alarm("init_sound_close.wav")

#6. 비정상 상태 판단값(result)에 따른 Motor/LED 구동 Arduino 제어부
def controlMotor(ain):
    if(ain==0):
        MT.write(b'1')
    elif(ain==1):
        MT.write(b'0')
    else:
        MT.write(b'0')

#7. 비정상 상태 판단값(result)에 따른 Bluetooth 통신 Arduino 제어부
def sendingBT(ain):
    if(ain==0):
        BT.write(b'1')
    elif(ain==1):
        BT.write(b'0')
    else:
        BT.write(b'0')

# Motor/LED 및 Bluetooth 통신 통합 제어부(using Multithread)
def control_func(av):
    tM = Thread(target=controlMotor, args=(av,))
    tB = Thread(target=sendingBT, args=(av,))
    tM.start()
    tB.start()
    tM.join()
    tB.join()

#####################################################################################################################

# <초기 세팅 session>

#1. EAR 측정을 위한 Global Variables 초기화
OPEN_EAR = 0 #For init_open_ear()
EAR_THRESH = 0 #Threashold value

#2. Variables for detecting if user is asleep.
#It doesn't matter what you use instead of a consecutive frame to check out drowsiness state. (ex. timer)
EAR_CONSEC_FRAMES = 20 
COUNTER = 0 #Frames counter.

#3. When the alarm rings, measure the time eyes are being closed.
closed_eyes_time = [] #The time eyes were being offed.
TIMER_FLAG = False #Flag to activate 'start_closing' variable, which measures the eyes closing time.
ALARM_FLAG = False #Flag to check if alarm has ever been triggered.

#4. When the alarm is rang, count the number of times it is rang, and prevent the alarm from ringing continuously.
ALARM_COUNT = 0 #Number of times the total alarm rang.
RUNNING_TIME = 0 #Variable to prevent alarm going off continuously.

#5. We should count the time eyes are being opened for data labeling.
PREV_TERM = 0 #Variable to measure the time eyes were being opened until the alarm rang.

#6. Variables for trained data generation and calculation fps.
#6. make trained data 
np.random.seed(9)
power, nomal, short = mtd.start(25) #actually this three values aren't used now. (if you use this, you can do the plotting)
#The array the actual test data is placed.
test_data = []
#The array the actual labeld data of test data is placed.
result_data = []
#For calculate fps
prev_time = 0

#7. Detect face & eyes.
print("Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#8. Run the cam.
print("Starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

#9. Threads to run the functions in which determine the EAR_THRESH. 
th_open = Thread(target = init_open_ear)
th_open.deamon = True
th_open.start()
th_close = Thread(target = init_close_ear)
th_close.deamon = True
th_close.start()

#####################################################################################################################

# <main 실행문>

main_func(1)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width = 1000)
    
    # Preprocessing(전처리 과정 - 조명 영향 최소화)
    L, gray = lr.light_removing(frame)
    
    rects = detector(gray,0)
    
    #checking fps. If you want to check fps, just uncomment below two lines.
    #prev_time, fps = check_fps(prev_time)
    #cv2.putText(frame, "fps : {:.2f}".format(fps), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        #(leftEAR + rightEAR) / 2 => both_ear. 
        both_ear = (leftEAR + rightEAR) * 800  #I multiplied by 1000 to enlarge the scope.

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)

        if both_ear < EAR_THRESH :
            if not TIMER_FLAG:
                start_closing = timeit.default_timer()
                TIMER_FLAG = True
            COUNTER += 1

            if COUNTER >= EAR_CONSEC_FRAMES:

                mid_closing = timeit.default_timer()
                closing_time = round((mid_closing-start_closing),3)

                if closing_time >= RUNNING_TIME:
                    if RUNNING_TIME == 0 :
                        CUR_TERM = timeit.default_timer()
                        OPENED_EYES_TIME = round((CUR_TERM - PREV_TERM),3)
                        PREV_TERM = CUR_TERM
                        RUNNING_TIME = 1.75

                    RUNNING_TIME += 2
                    ALARM_FLAG = True
                    ALARM_COUNT += 1

                    print("{0}st ALARM".format(ALARM_COUNT))
                    print("The time eyes is being opened before the alarm went off :", OPENED_EYES_TIME)
                    print("closing time :", closing_time)
                    test_data.append([OPENED_EYES_TIME, round(closing_time*10,3)])
                    result = mtd.run([OPENED_EYES_TIME, closing_time*10], power, nomal, short)  # abnormal state 판단(result = state of driver)
                    result_data.append(result)
                    t1 = Thread(target = alarm.select_alarm, args = (result, ))  # abnormal state 단계에 따른 Alarm 실행
                    t1.deamon = True
                    t1.start()
                    control_func(result)  # abnormal state 단계에 따른 Arduino control(MOTOR/LED, BLUETOOTH)

        else :
            COUNTER = 0
            TIMER_FLAG = False
            RUNNING_TIME = 0

            if ALARM_FLAG :
                end_closing = timeit.default_timer()
                closed_eyes_time.append(round((end_closing-start_closing),3))
                print("The time eyes were being offed :", closed_eyes_time)

            ALARM_FLAG = False

        cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (300,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
