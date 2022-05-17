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



def record_dual(vid_file, max_frames=10, num_cams=4, frame_pause=0):
    # Initializing dict to hold each image queue (from each camera)
    image_queue_dict = {}
    cams = [Camera(i, lock=True) for i in range(num_cams)]

    for c in cams:
        c.init()

        c.PixelFormat = "BayerRG8"  # BGR8 Mono8
        # c.BinningHorizontal = 1
        # c.BinningVertical = 1

        if False:
            c.GainAuto = 'Continuous'
            c.ExposureAuto = 'Continuous'
            # c.IspEnable = True

        c.GevSCPSPacketSize = 9000
        if num_cams > 2:
            c.DeviceLinkThroughputLimit = 85000000
            c.GevSCPD = 25000
        else:
            c.DeviceLinkThroughputLimit = 125000000
            c.GevSCPD = 25000
        # c.StreamPacketResendEnable = True

        # Initializing an image queue for each camera
        image_queue_dict[c.DeviceSerialNumber] = Queue(max_frames)
        dummy_queue = Queue(max_frames)

        print(c.DeviceSerialNumber, c.PixelSize, c.PixelColorFilter, c.PixelFormat,
              c.Width, c.Height, c.WidthMax, c.HeightMax, c.BinningHorizontal, c.BinningVertical)

    cams.sort(key=lambda x: x.DeviceSerialNumber)

    # print(cams[0].get_info('PixelFormat'))
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
            for _ in tqdm(range(max_frames)):

                if frame_pause > 0:
                    time.sleep(frame_pause)

                # get the image raw data
                # for each camera, get the current frame and assign it to
                # the corresponding camera
                real_times = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                for c in cams:
                    im = c.get_image()
                    timestamps = im.GetTimeStamp()

                    # get the data array
                    # Using try/except to handle frame tearing
                    try:
                        im = im.GetNDArray()
                    except Exception as e:
                        print(e)
                        tqdm.write('Bad frame')
                        continue

                    # Writing the frame information for the current camera to its queue
                    image_queue_dict[c.DeviceSerialNumber].put({'im': im, 'real_times': real_times, 'timestamps': timestamps})

        except KeyboardInterrupt:
            tqdm.write('Crtl-C detected')

        for c in cams:
            c.stop()

            image_queue_dict[c.DeviceSerialNumber].put(None)

    serials = [c.DeviceSerialNumber for c in cams]


    def write_queue(vid_file, image_queue, json_queue, serial):
        now = datetime.now()
        time_str = now.strftime('%Y%m%d_%H%M%S')
        vid_file = os.path.splitext(vid_file)[0] + f"_{serial}_{time_str}.mp4"

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

        # Adding the json info corresponding to the current camera to its own queue
        json_queue.put({'serial': serial, 'timestamps': timestamps, 'real_times': real_times, 'time_str': time_str})

        # average frame time from ns to s
        ts = np.asarray(timestamps)
        delta = np.mean(np.diff(ts, axis=0)) * 1e-9
        fps = 1.0 / delta

        print(f'Finished writing images. Final fps: {fps}')

        # indicate the last None event is handled
        image_queue.task_done()

    # initializing dictionary to hold json queue for each camera
    json_queue = {}
    # Start a writing thread for each camera
    for c in cams:
        serial = c.DeviceSerialNumber
        # Initializing queue to store json info for each camera
        json_queue[c.DeviceSerialNumber] = Queue(max_frames)
        threading.Thread(target=write_queue,
                         kwargs={'vid_file': vid_file, 'image_queue': image_queue_dict[serial],
                                 'json_queue': json_queue[c.DeviceSerialNumber], 'serial': serial}).start()

    acquire()

    # Joining the image queues for each camera
    # to allow each queue to be processed before moving on
    for c in cams:
        image_queue_dict[c.DeviceSerialNumber].join()

    # Creating a dictionary to hold the contents of each camera's json queue
    output_json = {}
    all_json = {}

    for j in json_queue:
        time_str = json_queue[j].queue[0]['time_str']
        real_times = json_queue[j].queue[0]['real_times']

        all_json[json_queue[j].queue[0]['serial']] = json_queue[j].queue[0]

    # defining the filename for the json file
    json_file = os.path.splitext(vid_file)[0] + f"_{time_str}.json"

    # combining the json information from each camera's queue
    all_serials = [all_json[key]['serial'] for key in all_json]
    all_timestamps = [all_json[key]['timestamps'] for key in all_json]

    output_json['serials'] = all_serials
    output_json['timestamps'] = [list(t) for t in zip(*all_timestamps)]
    output_json['real_times'] = real_times

    # writing the json file for the current recording session
    json.dump(output_json, open(json_file, 'w'))

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Record video from GigE FLIR cameras')
    parser.add_argument('vid_file', help='Video file to write')
    parser.add_argument('-m', '--max_frames', type=int, default=500, help='Maximum frames to record')
    args = parser.parse_args()

    record_dual(vid_file=args.vid_file, max_frames=args.max_frames)
    print("Done")
