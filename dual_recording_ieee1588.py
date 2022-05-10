import PySpin
from simple_pyspin import Camera
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime
from queue import Queue
import threading
import curses
import json
import time
import cv2
import os


def record_dual(vid_file, max_frames=100, num_cams=4, frame_pause=0):

    image_queue = Queue(max_frames)

    cams = [Camera(i, lock=True) for i in range(num_cams)]
    
    for c in cams:
        c.init()
        
        c.PixelFormat = "BayerRG8"  # BGR8 Mono8        
        #c.BinningHorizontal = 1
        #c.BinningVertical = 1

        if False:
            c.GainAuto = 'Continuous'
            c.ExposureAuto = 'Continuous'
            #c.IspEnable = True

        c.GevSCPSPacketSize = 9000
        if num_cams > 2:
            c.DeviceLinkThroughputLimit = 85000000
            c.GevSCPD = 25000
        else:
            c.DeviceLinkThroughputLimit = 125000000
            c.GevSCPD = 25000
        #c.StreamPacketResendEnable = True

        print(c.DeviceSerialNumber, c.PixelSize, c.PixelColorFilter, c.PixelFormat, 
            c.Width, c.Height, c.WidthMax, c.HeightMax, c.BinningHorizontal, c.BinningVertical)

    cams.sort(key = lambda x: x.DeviceSerialNumber)

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

    def acquire():

        for c in cams:    
            c.start()

        try:
            #for _ in range(max_frames):  # 
            for _ in tqdm(range(max_frames)):

                if frame_pause > 0:
                    time.sleep(frame_pause)
                
                # get the image raw data
                im = [c.get_image() for c in cams]

                # pull out IEEE1558 timestamps
                timestamps = [x.GetTimeStamp() for x in im]
                real_times = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                # get the data array
                try:
                    im = np.concatenate([x.GetNDArray() for x in im], axis=0)
                except Exception as e:
                    tqdm.write('Bad frame')
                    continue

                image_queue.put({'im': im, 'real_times': real_times, 'timestamps': timestamps})

        except KeyboardInterrupt:
            tqdm.write('Crtl-C detected')

        for c in cams:
            c.stop()

        image_queue.put(None)

    serials = [c.DeviceSerialNumber for c in cams]

    def write_queue(vid_file=vid_file, image_queue=image_queue, serials=serials):
        now = datetime.now()
        time_str = now.strftime('%Y%m%d_%H%M%S')
        json_file = os.path.splitext(vid_file)[0] + f"_{time_str}.json"
        vid_file = os.path.splitext(vid_file)[0] + f"_{time_str}.mp4"

        print(vid_file)

        timestamps = []
        real_times = []

        out_video = None

        for frame in iter(image_queue.get, None):
            if frame is None:
                break
            
            timestamps.append(frame['timestamps'])
            real_times.append(frame['real_times'])

            im = frame['im']
            if pixel_format == 'BayerRG8':
                im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)

            # need to collect two frames to track the FPS
            if out_video is None and len(real_times) == 1:
                last_im = im

            elif out_video is None and len(real_times) > 1:

                ts = np.asarray(timestamps)
                delta = np.mean(np.diff(ts, axis=0)) * 1e-9
                fps = 1.0 / delta
                tqdm.write(f'Computed FPS: {fps}')

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_video = cv2.VideoWriter(vid_file, fourcc, fps, (im.shape[1], im.shape[0]))
                out_video.write(last_im)

            else:
                out_video.write(im)

            image_queue.task_done()

        out_video.release()

        json.dump({'serials': serials, 'timestamps': timestamps, 'real_times': real_times}, open(json_file, 'w'))

        # average frame time from ns to s
        ts = np.asarray(timestamps)
        delta = np.mean(np.diff(ts, axis=0)) * 1e-9
        fps = 1.0 / delta

        print(f'Finished writing images. Final fps: {fps}')

        # indicate the last None event is handled
        image_queue.task_done()

    threading.Thread(target=write_queue).start()

    acquire()

    image_queue.join()

    return

        
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description = 'Record video from GigE FLIR cameras')
    parser.add_argument('vid_file', help='Video file to write')
    parser.add_argument('-m', '--max_frames', type=int, default=10000, help='Maximum frames to record')
    args = parser.parse_args()

    record_dual(vid_file=args.vid_file, max_frames=args.max_frames)
