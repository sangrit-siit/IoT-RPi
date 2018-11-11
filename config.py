from enum import Enum
import os

# ImageAI
EXECUTION_PATH = os.getcwd()

RASPBERRY_PI_MODE = Enum('RASPBERRY_PI_MODE', 'ENTRANCE EXIT')

OPERATION_MODE = RASPBERRY_PI_MODE.ENTRANCE

MODEL_PATH = os.path.join(EXECUTION_PATH, 'models')

OUTPUT_FILE_PATH = os.path.join(EXECUTION_PATH, 'output')

FRAMES_PER_SECOND = 30

LOG_PROGRESS = True

RETURN_DETECTED_FRAME = True

MINIMUM_PERCENTAGE_PROBABILITY = 30

DISPLAY_PERCENTAGE_PROBABILITY = False

DISDPLAY_OBJECT_NAME = True

SAVE_DETECTED_VIDEO = True


# OpenCV
CLASSIFIER_PATH = os.path.join(EXECUTION_PATH, 'classifier')

SCALE_FACTOR = 1.1

MIN_NEIGHBORS = 1

PI_MODE = 'entrance'

API_SERVER_IP = 'http://35.198.217.40/api'

API_PATH = 'openalpr'

if __name__ == '__main__':
    print(dir())


