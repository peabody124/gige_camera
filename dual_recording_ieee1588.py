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
import time
import yappi


# @profile
def record_dual(vid_file, max_frames=10, num_cams=1, frame_pause=0):
    # image_queue = Queue(max_frames)
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

        # c.GevSCPSPacketSize = 9000
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

    # @profile
    def acquire():

        for c in cams:
            c.start()

        try:
            # for _ in range(max_frames):  #
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
                    image_queue_dict[c.DeviceSerialNumber].put({'im': im.GetNDArray(), 'real_times': real_times, 'timestamps': timestamps})
                    dummy_queue.put({'im': im.GetNDArray(), 'real_times': real_times, 'timestamps': timestamps})

                # im = [c.get_image() for c in cams]
                #
                # # pull out IEEE1558 timestamps
                # timestamps = [x.GetTimeStamp() for x in im]
                # real_times = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                #
                # # get the data array
                # try:
                #     # print(len(im))
                #     # print(im[0].GetNDArray().shape)
                #     # cv2.imshow("video_frame",im[0].GetNDArray())
                #     # cv2.waitKey(0)
                #     im = np.concatenate([x.GetNDArray() for x in im], axis=0)
                #     # print("SHAPE: ",im.shape)
                #     # print(type(im))
                #     # cv2.imshow("video_frame",im)
                #     # cv2.waitKey(0)
                # except Exception as e:
                #     print(e)
                #     tqdm.write('Bad frame')
                #     continue
                #
                # image_queue_dict[c.DeviceSerialNumber].put({'im': im, 'real_times': real_times, 'timestamps': timestamps})

        except KeyboardInterrupt:
            tqdm.write('Crtl-C detected')

        for c in cams:
            c.stop()

            image_queue_dict[c.DeviceSerialNumber].put(None)
            dummy_queue.put(None)

    serials = [c.DeviceSerialNumber for c in cams]

    # @yappi.profile(profile_builtins=True)
    def write_queue(vid_file, image_queue, serial):
        t0 = time.time()
        now = datetime.now()
        time_str = now.strftime('%Y%m%d_%H%M%S')
        json_file = os.path.splitext(vid_file)[0] + f"_{serial[0]}_{time_str}.json"
        vid_file = os.path.splitext(vid_file)[0] + f"_{serial[0]}_{time_str}.mp4"

        print(vid_file)

        timestamps = []
        real_times = []

        out_video = None
        cnt_loop = 0.
        cvt_time_sum = 0.
        write_time_sum = 0.
        t1 = time.time()
        for frame in iter(image_queue.get, None):
            cnt_loop += 1
            if frame is None:
                break
            timestamps.append(frame['timestamps'])
            real_times.append(frame['real_times'])

            im = frame['im']
            t2 = time.time()
            if pixel_format == 'BayerRG8':
                im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)
            t3 = time.time()
            # print("CVT COLOR IN LOOP",t3-t2)
            cvt_time_sum += (t3 - t2)
            t3a = time.time()
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
            t4a = time.time()
            write_time_sum += (t4a - t3a)
            # print("WRITING IN LOOP",t4a-t3a)
            t4b = time.time()
            image_queue.task_done()

        out_video.release()

        json.dump({'serial': serial, 'timestamps': timestamps, 'real_times': real_times}, open(json_file, 'w'))

        # average frame time from ns to s
        ts = np.asarray(timestamps)
        delta = np.mean(np.diff(ts, axis=0)) * 1e-9
        fps = 1.0 / delta
        t4 = time.time()
        print(cnt_loop)
        print(f'Finished writing images. Final fps: {fps}')
        print("START ", t1 - t0)
        print("START LOOP", t2 - t1)
        print("CVT COLOR ", cvt_time_sum, cvt_time_sum / cnt_loop)
        print("WRITING ", write_time_sum, write_time_sum / cnt_loop)
        # indicate the last None event is handled
        image_queue.task_done()

    # for c in cams:
    #     serial = c.DeviceSerialNumber
    #
    # dummy_queue = Queue(max_frames)
    #
    # for i in image_queue_dict[serial].queue:
    #     dummy_queue.put(i)
    #
    # print(image_queue_dict[serial])
    # print(dummy_queue)

    for c in cams:
        serial = c.DeviceSerialNumber
        threading.Thread(target=write_queue,
                         kwargs={'vid_file': vid_file, 'image_queue': image_queue_dict[serial], 'serial': [serial]},
                         daemon=True).start()
        threading.Thread(target=write_queue,
                         kwargs={'vid_file': vid_file+"2", 'image_queue': dummy_queue, 'serial': [serial]},
                         daemon=True).start()



    # threading.Thread(target=write_queue,
    #                  kwargs={'vid_file': vid_file, 'image_queue': image_queue_dict[serial], 'serial': [serial]},daemon=True).start()
    # threading.Thread(target=write_queue,
    #                  kwargs={'vid_file': vid_file + "2", 'image_queue': dummy_queue, 'serial': [serial]},daemon=True).start()

    acquire()


    image_queue_dict[c.DeviceSerialNumber].join()
    dummy_queue.join()

    # for c in cams:
    #     image_queue_dict[c.DeviceSerialNumber].join()

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Record video from GigE FLIR cameras')
    parser.add_argument('vid_file', help='Video file to write')
    parser.add_argument('-m', '--max_frames', type=int, default=500, help='Maximum frames to record')
    args = parser.parse_args()

    record_dual(vid_file=args.vid_file, max_frames=args.max_frames)
    print("Done")