import time

import cv2
import numpy as np
import serial
import math

port = '/dev/ttyACM0'
ser = serial.Serial(port, 57600)
ser.timeout = 0.01 

MinHeight = 300
MaxHeight = 310

old_R = 0
old_L = 0


def detect_blue_line(src):
    # HSV로 변환 후 파란색만 추출
    b_l_threshold = (100, 100, 70)
    b_h_threshold = (150, 255, 255)
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    b_mask = cv2.inRange(hsv_img, b_l_threshold, b_h_threshold)
    blue_img = cv2.bitwise_and(src, src, mask=b_mask)

    # Grayscale로 변환
    grayscale = cv2.cvtColor(blue_img, cv2.COLOR_BGR2GRAY)
    # 모서리 검출
    can = cv2.Canny(grayscale, 50, 200, None, 3)

    height = can.shape[0]
    rectangle = np.array([[(0, height), (120, 50), (520, 50), (640, height)]])
    mask = np.zeros_like(can)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(can, mask)
    ccan = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)

    mask = np.zeros_like(can)
    rectangle = np.array([[(0, MinHeight), (0, MaxHeight), (640, MinHeight), (640, MaxHeight)]])
    cv2.fillPoly(mask, rectangle, 255)
    roi = cv2.bitwise_and(grayscale, mask)
    # cv2.imshow("ccan", ccan)

    contours, hei = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = list()

    totalArea = 0

    for x in contours:
        totalArea += cv2.contourArea(x);
        objs.append(x);

    objs.sort(key=lambda x: cv2.contourArea(x))
    cx = 0.0
    cy = 0.0
    if len(contours) > 0:
        c0 = contours[-1];
        leftMost = tuple(c0[c0[:, :, 0].argmin()][0])
        rightMost = tuple(c0[c0[:, :, 0].argmax()][0])
        topMost = tuple(c0[c0[:, :, 1].argmin()][0])
        bottomMost = tuple(c0[c0[:, :, 1].argmax()][0])
    
        cx = (int)((leftMost[0] + rightMost[0]) / 2);
        cy = (int)((topMost[1] + bottomMost[1]) / 2);

        cv2.circle(roi, (cx, cy), 2, (255, 255, 255))
    cv2.imshow("roi", roi)


    # 원본에 합성
    outcome = cv2.addWeighted(src, 1, ccan, 1, 0)
    return outcome, cx, cy, totalArea


cam = cv2.VideoCapture(0)
frame_rate = 10

if not cam.isOpened():
    print("Could not open webcam")
    exit()

st = 0
lll=0
vvals=[10,10]
cam.set(cv2.CAP_PROP_FPS, frame_rate)
while cam.isOpened():
    status, frame = cam.read()
    frame = cv2.flip(frame, 0)
    vals = [170, 170]
    
    if status:
        frame = cv2.resize(frame, (640, 360))

        f, cx, cy, ta = detect_blue_line(frame)  # f = img, cx = dot_pos, cy = not use(line), ta = total area

        cv2.imshow('test', f)
        

        # if total area > 1000
        print("\n")
        print("cx: " + str(cx) + " / cy: " + str(cy) + " / ta: " + str(ta))
        if ta > 1200:
            if st == 1:
                vals[0] = vvals[0]
                vals[1] = vvals[1]
                ser.write(b'M')
                ser.write(bytes(bytearray(vals)))
            else:
                data = b''
                if vvals[0] != 0:
                    vvals[0] = 0
                    vvals[1] = 0
                    ser.write(b'M')
                    ser.write(bytes(bytearray(vvals)))
                if lll==0:
                    ser.write(b'L')
                    while(data!=b'O'):
                        try:
                            data = ser.read(1)  
                        except ser.SerialTimeoutException:
                            print("Timeout occurred, no data available")
                
                    lll=1
                else:
                    ser.write(b'U')
                    while(data!=b'X'):
                        try:
                            data = ser.read(1)  
                        except ser.SerialTimeoutException:
                            print("Timeout occurred, no data available")
                    lll=0
                vvals[0] = old_R
                vvals[1] = old_L
                ser.write(b'M')
                ser.write(bytes(bytearray(vvals)))
                st = 1
        
        elif cx == 0 and cy == 0:
            vals[0] = old_R
            vals[1] = old_L

            ser.write(b'M')
            ser.write(bytes(bytearray(vals)))
            print(bytes(bytearray(vals)))
            print("\n")
            st = 0
        else:
            # move
            weight = 0.4

            vals[0] = vals[0] + int((cx - 320) * weight)
            vals[1] = vals[1] - int((cx - 320) * weight)
            print("vals[0] = " + str(vals[0]) + "   vals[1] = ", str(vals[1]))
            # print("\n")

            if (vals[0] < 0):
                vals[0] = 0
            elif (vals[0] > 255):
                vals[0] = 255
            if (vals[1] < 100):
                vals[1] = 0
            elif (vals[1] > 255):
                vals[1] = 255

            old_R = vals[0]
            old_L = vals[1]

            ser.write(b'M')
            ser.write(bytes(bytearray(vals)))
            print(bytes(bytearray(vals)))
            print("\n")
            st = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        ser.close()
        break


cv2.destroyAllWindows()
