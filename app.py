from imageai.Detection import VideoObjectDetection
from matplotlib import pyplot as plt
from config import *

import cv2
import os

def forFrame(frame_number, output_array, output_count, returned_frame):
    rgb_frame = cv2.cvtColor(returned_frame, cv2.COLOR_BGR2RGB)
    plt.clf()
    plt.imshow(rgb_frame, interpolation="none")
    plt.pause(0.01)

    print("FOR FRAME " , frame_number)

    print("Output for each object : ")
    for output in output_array:
        print('\t {}'.format(output))

    print("Output count for unique objects : ")
    for output in output_count:
        print('\t {}: {}'.format(output, output_count[output]))

    print("------------END OF A FRAME --------------")


def getCar(output_array, output_count):
    car = [output for output in output_array if output['name']=='car']
    return car



if __name__ == '__main__':
    print('OPERATION_MODE:\t{}'.format(OPERATION_MODE))
    print('FRAMES_PER_SECOND:\t{}'.format(FRAMES_PER_SECOND))

    camera = cv2.VideoCapture(0)

    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(MODEL_PATH, "yolo.h5"))
    detector.loadModel()

    plt.show()


    video_path = detector.detectObjectsFromVideo(
        camera_input=camera,
        output_file_path=os.path.join(EXECUTION_PATH,'camera_detected_video'),
        frames_per_second=FRAMES_PER_SECOND,
        log_progress=LOG_PROGRESS,
        minimum_percentage_probability=MINIMUM_PERCENTAGE_PROBABILITY,
        return_detected_frame = RETURN_DETECTED_FRAME,
        per_frame_function=forFrame,
    )

    print(video_path)
