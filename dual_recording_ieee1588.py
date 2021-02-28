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
print(f'Cams: {cams}')

for c in cams:
    c.init()
    
    #c.PixelFormat = "BayerRG8"
    #c.PixelFormat = "Mono8"

    #c.GainAuto = 'Continuous'
    #c.ExposureAuto = 'Continuous'
    #c.IspEnable = True
    
    c.BinningHorizontal = 2
    c.BinningVertical = 2
    #c.Width = c.SensorWidth
    #c.Height = c.SensorHeight

    c.DeviceLinkThroughputLimit = 100000000
    c.GevSCPSPacketSize = 9000
    c.GevSCPD = 72000    

    print(c.DeviceSerialNumber, c.PixelSize, c.PixelColorFilter, c.PixelFormat, 
          c.Width, c.Height, c.WidthMax, c.HeightMax, c.BinningHorizontal, c.BinningVertical)

print(cams[0].get_info('PixelFormat'))

for c in cams:    
    c.GevIEEE1588 = True

time.sleep(10)

for c in cams:
    c.GevIEEE1588DataSetLatch()
    print(c.GevIEEE1588StatusLatched)
    print(c.GevIEEE1588OffsetFromMasterLatched)

for c in cams:    
    c.start()
    
images = []
timestamps = []
print('Acquiring images')
for i in tqdm(range(frames)):
    time.sleep(0.3)
    #time.sleep(2)
    
    if False:
        im = np.concatenate([c.get_array() for c in cams], axis=0)
        timestamps.append(datetime.now().timestamp())
    else:
        im = [c.get_image() for c in cams]

        timestamps.append([x.GetTimeStamp() for x in im])
    
        #print(dir(im[0]))
        for i, x in enumerate(im):
            if not x.IsValid():
                print(f'{i} Valid: {x.IsValid()}. Timestamp: {x.GetTimeStamp()}. Num Channels: {x.GetNumChannels()}. Size: {x.GetWidth()} x {x.GetHeight()}. Pixel format: {x.GetPixelFormat()} {x.GetPixelFormatName()}')
    
        #.Convert(PySpin.PixelFormat_BayerRG8, PySpin.HQ_LINEAR)
        im = np.concatenate([x.GetNDArray() for x in im], axis=0)
        
    images.append(im)

for c in cams:
    c.stop()

json.dump(timestamps, open(os.path.join(output_dir, 'timestamps.json'), 'w'))

# average frame time from ns to s
ts = np.asarray(timestamps)
delta = np.mean(np.diff(ts, axis=0)) * 1e-9
fps = 1.0 / delta

#print(np.diff(ts, axis=0))
#print(np.diff(ts, axis=1))

print(f'Writing images. Computed fps: {fps}')
out_video = None
for i, im in tqdm(enumerate(images)):
    
    im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)
    Image.fromarray(im[..., ::-1]).save(os.path.join(output_dir, '%08d.png' % i))
    
    #if out_video is None:
    #    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #    out_video = cv2.VideoWriter(os.path.join(output_dir, 'vid.mp4'), fourcc, fps, (im.shape[1], im.shape[0]))
    #out_video.write(im)

#out_video.release()
    

    
