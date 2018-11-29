import cv2
import os
import requests
import time

from picamera.array import PiRGBArray
from picamera import PiCamera

from config import *

def carClassifier(frame, classifier, scaleFactor, minNeighbors, flags=0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return classifier(gray, scaleFactor, minNeighbors, flags)

##################################################################################################
def carDetection(image, cars):

    #cars = [item for item in detection if item['item'] == 'car']
    if len(cars)>0:
        print(cars)
        cv2.imwrite('output.jpg', image)
        image = open('output.jpg','rb')
        files = {"image":image}
        data = {"action": PI_MODE}
        # post to server
        dest = '/'.join([API_SERVER_IP, API_PATH])
        response = requests.post(dest, files=files, data=data)
        if response.content.decode('utf-8').strip() == 'OK':
            print('OK')
            time.sleep(3)
    #end if

##################################################################################################
if __name__ == '__main__':
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    #camera.resolution = (640, 480)
    camera.framerate = 32
    #rawCapture = PiRGBArray(camera, size=(640, 480))
    rawCapture = PiRGBArray(camera)
    
    classifier = cv2.CascadeClassifier(os.path.join(CLASSIFIER_PATH,'cars.xml'))
    # allow the camera to warmup
    time.sleep(0.1)
     
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            image = frame.array
            #cv2.imshow("Frame", image)
            cars = carClassifier(image, classifier.detectMultiScale, 1.1, 1)
            carDetection(image, cars)
            # show the frame
            
            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)
            
            #key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            #if key == ord("q"):
            #    break
