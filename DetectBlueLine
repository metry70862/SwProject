import cv2
import numpy as np
import os, sys
import serial
import time

# 시리얼 장치 연결
# 보레이드 9600, 장치 위치 ttyUSB0
ser = serial.Serial('/dev/ttyACM0',9600, timeout = 5)

def run():
    #시리얼 쓰기
    ser.write(bytes(bytearray([0x01,0x02,0x03,0x04,0x05])))
    while True:
        # 시리얼 읽기 (5바이트씩 읽음)
        line = ser.read(5)
        if len(line) == 0:
                break;
        # 헥사 코드로 출력
        hex_list = ["{:x}".format(ord(c)) for c in line];
        print ''.join(hex_list)
while True:
    run();

def DetectBlueLine(src):

    #HSV로 변환후 파란색만 추출
    b_l_threshold = (100, 100, 100)
    b_h_threshold = (150, 255, 255)
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    b_mask = cv2.inRange(hsv_img, b_l_threshold, b_h_threshold)
    blue_img = cv2.bitwise_and(src, src, mask=b_mask)

    # GrayScale 로 변환
    
    grayscale = cv2.cvtColor(blue_img, cv2.COLOR_BGR2GRAY)

    # 모서리 검출
    can = cv2.Canny(grayscale, 50, 200, None, 3)

    height = can.shape[0]
    rectangle = np.array([[(0, height), (120, 200), (520, 200), (640, height)]])
    mask = np.zeros_like(can)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(can, mask)
    ccan = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)

    line_arr = cv2.HoughLinesP(masked_image, 1, np.pi / 180, 20, minLineLength=10, maxLineGap=75)

    # 중앙을 기준으로 오른쪽, 왼쪽 직선 분리
    line_R = np.empty((0, 5), int)
    line_L = np.empty((0, 5), int)
    if line_arr is not None:
        line_arr2 = np.empty((len(line_arr), 5), int)
        for i in range(0, len(line_arr)):
            l = line_arr[i][0]
            line_arr2[i] = np.append(line_arr[i], np.array((np.arctan2(l[1] - l[3], l[0] - l[2]) * 180) / np.pi))
            if line_arr2[i][1] > line_arr2[i][3]:
                temp = line_arr2[i][0], line_arr2[i][1]
                line_arr2[i][0], line_arr2[i][1] = line_arr2[i][2], line_arr2[i][3]
                line_arr2[i][2], line_arr2[i][3] = temp
            if line_arr2[i][0] < 320 and (170 > abs(line_arr2[i][4]) > 95):
                line_L = np.append(line_L, line_arr2[i])
            elif line_arr2[i][0] > 320 and (170 > abs(line_arr2[i][4]) > 95):
                line_R = np.append(line_R, line_arr2[i])
    line_L = line_L.reshape(int(len(line_L) / 5), 5)
    line_R = line_R.reshape(int(len(line_R) / 5), 5)

    # 중앙과 가까운 오른쪽, 왼쪽 선을 최종 차선으로 인식
    try:
        line_L = line_L[line_L[:, 0].argsort()[-1]]
        degree_L = line_L[4]
        cv2.line(ccan, (line_L[0], line_L[1]), (line_L[2], line_L[3]), (255, 255, 255), 3, cv2.LINE_AA)
    except:
        degree_L = 0
    try:
        line_R = line_R[line_R[:, 0].argsort()[0]]
        degree_R = line_R[4]
        cv2.line(ccan, (line_R[0], line_R[1]), (line_R[2], line_R[3]), (255, 255, 255), 3, cv2.LINE_AA)
    except:
        degree_R = 0

    # 원본에 합성
    outcome = cv2.addWeighted(src, 1, ccan, 1, 0)
    return outcome, degree_L, degree_R


cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Could not open webcam")
    exit()

while cam.isOpened():

    status, frame = cam.read()

    if status:
        frame = cv2.resize(frame, (640, 360))
        cv2.imshow('test', DetectBlueLine(frame)[0])
        l, r = DetectBlueLine(frame)[1], DetectBlueLine(frame)[2]

        if abs(l) <= 155 or abs(r) <= 155:
            if l == 0 or r == 0:
                if l < 0 or r < 0:
                    print('left')
                elif l > 0 or r > 0:
                    print('right')
            elif abs(l - 15) > abs(r):
                print('right')
            elif abs(r + 15) > abs(l):
                print('left')
            else:
                print('go')
        else:
            if l > 155 or r > 155:
                print('hard right')
            elif l < -155 or r < -155:
                print('hard left')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
