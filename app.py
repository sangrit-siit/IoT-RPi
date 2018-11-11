import cv2
import os

from config import *

def carClassifier(frame, classifier, scaleFactor, minNeighbors, flags=0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return classifier(gray, scaleFactor, minNeighbors, flags)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    classifier = cv2.CascadeClassifier(os.path.join(CLASSIFIER_PATH,'cars.xml'))

    while True:
        ret, frames = cap.read()
        cars = carClassifier(frames, classifier.detectMultiScale, 1.1, 1)

        for (x,y,w,h) in cars:
            cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)

        # Display frames in a window
        cv2.imshow('video', frames)

        # Wait for Esc key to stop
        if cv2.waitKey(33) == 27:
            break

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()
