import numpy as np
import cv2
import json
import datetime
import time
from threading import Thread
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import base64
import os
import requests
import time
cls = lambda: os.system('cls')


with open('config.json') as file:
    config = json.load(file)

#cap1 = cv2.VideoCapture("rtsp://25.169.165.35:8080/h264_pcm.sdp")
#cap1 = cv2.VideoCapture(0)
#cap1 = cv2.VideoCapture("footage1.mp4")
cap1 = cv2.VideoCapture("tes.mp4")

counter = 1
def telesend(text):
    url = "https://api.telegram.org/token bot/sendMessage?chat_id=your telegram id&parse_mode=Markdown&text=" + text
    requests.get(url)

net = cv2.dnn.readNetFromCaffe(config["object_model"], config["object_path"])
ret, frame = cap1.read()
(h, w) = frame.shape[:2]
text = "Unoccupied"
exit = False
email = True
time_format = "%d-%m-%Y %H:%M:%S"
polygon = Polygon(eval(config["points"]))
while True:
    try:
        if exit is False:

            ret, image = cap1.read()

            if image is None:
                break

            ori_image = image.copy()
            (h, w) = image.shape[:2]
            if cap1.get(1) % config["in_fps"] == 0:
                if text == "Unoccupied":
                    taken = False
                    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
                    net.setInput(blob)
                    detections = net.forward()
                    threshold = 0.6
                    for i in np.arange(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > threshold:
                            idx = int(detections[0, 0, i, 1])

                            if idx == 15:
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")
                                center = (endX,endY)
                                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                                #telesend("Test")
                                #cls()
                                point = Point(center)
                                #print(point)
                                counter += 1
                                if polygon.contains(point) == True:
                                   if counter % 5 == 0:
                                        telesend("Penyusup Terdeteksi")

            cv2.polylines(image,np.int32([eval(config["points"])]),True,(255,0,0),2)
            cv2.putText(image, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.namedWindow("intrusion", cv2.WINDOW_NORMAL)
            cv2.imshow("intrusion", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Don't close the thread until it finished")
                exit = True
    except Exception as E:
        print(E)
        print("Don't close the thread until it finished")
        cap1.release()
        cap1 = cv2.VideoCapture(config["camera_url"])
        #exit = True

cap1.release()
cv2.destroyAllWindows()

print("Selesai")

