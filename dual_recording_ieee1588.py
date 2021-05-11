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


def record_dual(vid_file, max_frames=100, num_cams=2, frame_pause=0):
    
    cams = [Camera(i, lock=True) for i in range(num_cams)]
    print(f'Cams: {cams}')

    for c in cams:
        c.init()
        
        c.PixelFormat = "BayerRG8"  # BGR8 Mono8        
        c.BinningHorizontal = 1
        c.BinningVertical = 1

        if False:
            c.GainAuto = 'Continuous'
            c.ExposureAuto = 'Continuous'
            #c.IspEnable = True

        c.DeviceLinkThroughputLimit = 100000000
        c.GevSCPSPacketSize = 9000
        c.GevSCPD = 72000    

        print(c.DeviceSerialNumber, c.PixelSize, c.PixelColorFilter, c.PixelFormat, 
            c.Width, c.Height, c.WidthMax, c.HeightMax, c.BinningHorizontal, c.BinningVertical)

    #print(cams[0].get_info('PixelFormat'))
    pixel_format = cams[0].PixelFormat

    if not all([c.GevIEEE1588]):
        print('Cameras not synchronized. Enabling IEEE1588 (takes 10 seconds)')
        for c in cams:    
            c.GevIEEE1588 = True

        time.sleep(10)

    for c in cams:
        c.GevIEEE1588DataSetLatch()
        print(c.GevIEEE1588StatusLatched, c.GevIEEE1588OffsetFromMasterLatched)

    def acquire(win):
        images = []
        timestamps = []
        real_times = []

        win.nodelay(True)
        win.clear()
        win.addstr('Acquiring images. Press enter to stop')
  
        for c in cams:    
            c.start()

        for _ in range(max_frames):  # tqdm(range(max_frames)):

            try:
                key = win.getkey()
                if key == os.linesep:
                    print('Key detected')
                    break
            except Exception:
                pass

            if frame_pause > 0:
                time.sleep(frame_pause)
            
            # get the image raw data
            im = [c.get_image() for c in cams]

            # pull out IEEE1558 timestamps
            timestamps.append([x.GetTimeStamp() for x in im])
            real_times.append(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
        
            # get the data array
            im = np.concatenate([x.GetNDArray() for x in im], axis=0)        

            images.append(im)

        for c in cams:
            c.stop()

        return images, timestamps, real_times

    now = datetime.now()
    time_str = now.strftime('%Y%m%d_%H%M%S')
    json_file = os.path.splitext(vid_file)[0] + f"_{time_str}.json"
    vid_file = os.path.splitext(vid_file)[0] + f"_{time_str}.mp4"

    import curses
    images, timestamps, real_times = curses.wrapper(acquire)
    im = images[0]

    json.dump({'timestamps': timestamps, 'real_times': real_times}, open(json_file, 'w'))

    # average frame time from ns to s
    ts = np.asarray(timestamps)
    print(ts)
    delta = np.mean(np.diff(ts[:, :-1], axis=0)) * 1e-9
    fps = 1.0 / delta

    #print(np.diff(ts, axis=0))
    #print(np.diff(ts, axis=1))

    print(f'Writing images. Computed fps: {fps}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(vid_file, fourcc, fps, (im.shape[1], im.shape[0]))

    for i, im in tqdm(enumerate(images)):
        
        #if pixel_format == 'BGR8':
        #    im = im[..., ::-1]

        if pixel_format == 'BayerRG8':
            im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)
        out_video.write(im)

        # very slow
        # Image.fromarray(im[..., ::-1]).save(os.path.join(output_dir, '%08d.png' % i))

    out_video.release()
        

        
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description = 'Record video from GigE FLIR cameras')
    parser.add_argument('vid_file', help='Video file to write')
    parser.add_argument('-m', '--max_frames', type=int, default=10000, help='Maximum frames to record')
    args = parser.parse_args()

    record_dual(vid_file=args.vid_file, max_frames=args.max_frames)
