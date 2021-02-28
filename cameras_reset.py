import PySpin
from simple_pyspin import Camera
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json
import time
import cv2
import os

frames = 20
output_dir = 'cal_vid'
num_cals = 2
cams = [Camera(i, lock=True) for i in range(num_cals)]
#cams = [Camera(i, lock=True) for i in [1]]
#cams = cams[:2]

print(f'List of cams: {cams}')

for c in cams:
    c.init()
    print('Reset')
    c.DeviceReset()
