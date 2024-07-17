
import sys
import logging
from core import NanonisController
from modules.MotorOperation import ChangeArea


if __name__ == '__main__':
    direction_list = ['X+', 'X-', 'Y+', 'Y-']
    direction = 'Y-'
    
    if len(sys.argv) > 1:
        direction = sys.argv[1].upper()
        if direction not in direction_list:
            logging.warn('Invalid direction. Using Y-')
            direction = 'Y-'

    nanonis = NanonisController()
    ChangeArea(nanonis, direction=direction).do()

