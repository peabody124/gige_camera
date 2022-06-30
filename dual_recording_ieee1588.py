import PySpin
from simple_pyspin import Camera
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime
from queue import Full,Empty
from queue import Queue
import threading
import curses
import json
import time
import cv2
import os
import _thread
import pynput

# Defining window size based on number
# of cameras(key)
window_sizes = {1: np.array([1, 1]),
                2: np.array([1, 2]),
                3: np.array([2, 2]),
                4: np.array([2, 2]),
                5: np.array([2, 3]),
                6: np.array([2, 3])
                }

def record_dual(vid_file, max_frames=100, num_cams=1, frame_pause=0, preview = True):
    # Initializing dict to hold each image queue (from each camera)
    image_queue_dict = {}
    if preview:
        visualization_queue = Queue(1)
        out_queue = Queue(1)
        e = threading.Event() 
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
                size_flag = 0
                real_time_images = []
                for c in cams:
                    im = c.get_image()
                    timestamps = im.GetTimeStamp()

                    # get the data array
                    # Using try/except to handle frame tearing
                    try:
                        im = im.GetNDArray()

                        # if preview is enabled, save the size of the first image
                        # and append the image from each camera to a list
                        if preview:
                            real_time_images.append(im)
                            if size_flag == 0:
                                size_flag = 1
                                image_size = im.shape
                    except Exception as e:
                        tqdm.write('Bad frame')
                        continue

                    # Writing the frame information for the current camera to its queue
                    image_queue_dict[c.DeviceSerialNumber].put({'im': im, 'real_times': real_times, 'timestamps': timestamps})

                if preview:
                    t0 = time.time()
                    if len(real_time_images) < np.prod(window_sizes[num_cams]):
                        # Add extra square to fill in empty space if there are
                        # not enough images to fit the current grid size
                        real_time_images.extend([np.zeros_like(real_time_images[0]) for i in range(np.prod(window_sizes[num_cams]) - len(real_time_images))])

                    desired_width = image_size[1]
                    desired_height = image_size[0]

                    # create output visualization shape
                    desired_zeros = np.zeros_like(real_time_images[0])
                    im_window = np.zeros_like(desired_zeros,shape=np.array(desired_zeros.shape) * window_sizes[num_cams])

                    # removing padding code for now, making assumption that all cameras
                    # will have same sized images

                    im_counter = 0
                    w_offset = 0
                    h_offset = 0
                    for r in range(window_sizes[num_cams][0]):
                        for c in range(window_sizes[num_cams][1]):

                            im_window[h_offset:h_offset+desired_height,w_offset:w_offset+desired_width] = real_time_images[im_counter]
                            im_counter += 1
                            w_offset += desired_width

                        h_offset += desired_height
                        w_offset = 0
                    t1 = time.time()
                    # Add combined image to queue
                    if visualization_queue.empty():
                        #print("in try")
                        #print(visualization_queue.empty())
                        t2 = time.time()
                        visualization_queue.put({'im': im_window},block=False)
                        #visualization_queue.put(im_window,block=False)
                        #cv2.imshow("test",im_window)
                    else:
                        #if e.is_set():
                        #    _thread.interrupt_main()
                        t3 = time.time()
                        #cur_im = visualization_queue.get()
                        t4 = time.time()
                        #cv2.imshow("Preview", cv2.cvtColor(cur_im, cv2.COLOR_BAYER_RG2RGB))
                        t5 = time.time()
                        #g = cv2.waitKey(1)
                        t6 = time.time()


                        #print("T1",t1-t0)
                        #print("T2",t2-t1)
                        #print("T3",t3-t2)
                        #print("T4",t4-t3)
                        #print("T5",t5-t4)
                        #print("T6",t6-t5)
                    #    cv2.destroyAllWindows()
                    #    if g == ord('q') or g == ord('c'):
                    #        _thread.interrupt_main()
                    #except Full:
                    #    pass       
                    #    print("in full except")
                    #    #print(visualization_queue.empty())
                    #
                    #    #for frame in iter(visualization_queue.get, None):
                    #    #print(visualization_queue.empty())
                    #    #cv2.imshow("Preview", cv2.cvtColor(frame['im'], cv2.COLOR_BAYER_RG2RGB))
                    #    #cv2.waitKey(1)
                    #
                    #    #print("%%%%%%%%%%"+str(cv2.waitKey(1)))
                    #except Empty:
                    #    print("in empty except")
                    # try:
                    #    #print("in try")
                    #    #print(visualization_queue.empty())
                    #    #visualization_queue.put({'im': im_window},block=False)
                    #    visualization_queue.put(im_window,block=False)
                    #    #cv2.imshow("test",im_window)
                    #except Full:    
                    #    cur_im = visualization_queue.get(block=False)
                    #    cv2.imshow("Preview", cv2.cvtColor(cur_im, cv2.COLOR_BAYER_RG2RGB))
                    #    g = cv2.waitKey(1)
                    #    #cv2.destroyAllWindows()
                    #    if g == ord('q') or g == ord('c'):
                    #        _thread.interrupt_main()
                    #
                    #    #print("in full except")
                    #    #print(visualization_queue.empty())
                    #
                    #    #for frame in iter(visualization_queue.get, None):
                    #    #print(visualization_queue.empty())
                    #    #cv2.imshow("Preview", cv2.cvtColor(frame['im'], cv2.COLOR_BAYER_RG2RGB))
                    #    #cv2.waitKey(1)
                    #
                    #    #print("%%%%%%%%%%"+str(cv2.waitKey(1)))
                    #except Empty:
                    #    print("in empty except")
                    #if visualization_queue.empty():
                    #    #print("in try")
                    #    #print(visualization_queue.empty())
                    #    #visualization_queue.put({'im': im_window},block=False)
                    #    visualization_queue.put(im_window,block=False)
                    #    #cv2.imshow("test",im_window)
                    #else:    
                    #    cur_im = visualization_queue.get(block=False)
                    #    cv2.imshow("Preview", cv2.cvtColor(cur_im, cv2.COLOR_BAYER_RG2RGB))
                    #    g = cv2.waitKey(1)
                    #    #cv2.destroyAllWindows()
                    #    if g == ord('q') or g == ord('c'):
                    #        _thread.interrupt_main()
                    #
                    #    #print("in full except")
                    #    #print(visualization_queue.empty())
                    #
                    #    #for frame in iter(visualization_queue.get, None):
                    #    #print(visualization_queue.empty())
                    #    #cv2.imshow("Preview", cv2.cvtColor(frame['im'], cv2.COLOR_BAYER_RG2RGB))
                    #    #cv2.waitKey(1)
                    #
                    #    #print("%%%%%%%%%%"+str(cv2.waitKey(1)))
                    #except Empty:
                    #    print("in empty except")


        except KeyboardInterrupt:
            tqdm.write('Crtl-C detected')

        for c in cams:
            c.stop()

            image_queue_dict[c.DeviceSerialNumber].put(None)

    serials = [c.DeviceSerialNumber for c in cams]

    def visualize(image_queue):
        for frame in iter(image_queue.get, None):
        #print(image_queue.get)
        #print(iter(image_queue.get, None))
        #print(image_queue.get())
        #frame = next(iter(image_queue.get, None))
            cv2.imshow("Preview", cv2.cvtColor(frame['im'], cv2.COLOR_BAYER_RG2RGB))
            cv2.waitKey(1)

        #if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('c'):
        #    _thread.interrupt_main()

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

    if preview:
        threading.Thread(target=visualize,kwargs={'image_queue':visualization_queue},daemon=visualize).start()
         
        def on_press(key):
            print('{0} pressed'.format(key))
            if key == pynput.keyboard.Key.esc or key == pynput.keyboard.KeyCode.from_char('q') or key == pynput.keyboard.KeyCode.from_char('c'):
                # Stop listener
                _thread.interrupt_main()
                return False

        def on_release(key):
            print('{0} release'.format(key))

        # Collect events until released
        listener = pynput.keyboard.Listener(on_press=on_press,on_release=on_release,suppress=True)
        listener.start()


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
    parser.add_argument('-m', '--max_frames', type=int, default=10000, help='Maximum frames to record')
    parser.add_argument('-n', '--num_cams', type=int, default=4, help='Number of input cameras')
    parser.add_argument('-f', '--frame_pause', type=int, default=0, help='Time to pause between frames of video')
    parser.add_argument('-p','--preview', default=True, action='store_true', help='Allow real-time visualization of video')
    parser.add_argument('--no-preview', dest='preview', action='store_false')
    args = parser.parse_args()

    record_dual(vid_file=args.vid_file, max_frames=args.max_frames, num_cams=args.num_cams,frame_pause=args.frame_pause,preview=args.preview)
