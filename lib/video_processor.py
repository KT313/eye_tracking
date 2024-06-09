from .imports import *
from . import misc

import math
import pyvirtualcam

image_eye_open = cv2.imread('eye_open.png', cv2.IMREAD_UNCHANGED)
image_eye_closed = cv2.imread('eye_closed_new.png', cv2.IMREAD_UNCHANGED)

class FrameElement():
    def __init__(self, face_bbox = None, eyes_points = None, frame = None, cutout_frame = None, stats = None, face_cutout_coords = None, detected_face = None, frame_id = None, annotate_eyes = None, annotate_face = None):
        self.frame_id = frame_id
        self.face_bbox = face_bbox
        self.eyes_points = eyes_points
        self.frame = frame
        self.cutout_frame = cutout_frame
        self.stats = stats
        self.face_cutout_coords = face_cutout_coords
        self.detected_face = detected_face
        self.annotate_eyes = annotate_eyes
        self.annotate_face = annotate_face

        self.processing_times = []

class VideoProcessor():
    def __init__(self, eye_model_size=None, form="onnx", eye_model="auto"):
        self.eye_model_size = eye_model_size
        self.avg_len = 60
        self.frame_counter = 0
        self.interpolate_rate = 1
        self.eyes_int_rate = 1
        self.iswebcam = False
        self.native_fps = 30
        self.model_face = YOLO('models/yolov8n-face-lindevs.onnx', task="detect")
        if eye_model != "auto":
            self.model_eyes = YOLO(eye_model, task="pose")
        elif self.eye_model_size != None:
            self.model_eyes = YOLO(f'models/eyes_{self.eye_model_size}_auto.{form}', task="pose")
        else:
            self.model_eyes = YOLO(f'models/eyes_256_auto.{form}', task="pose")
        self.buffer = {}
        self.last_output_time = time.time()
        self.last_process_time = time.time()
        self.stop = False
        self.frame_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.return_instead = False
        self.return_instead_merker = []

    def start(self, video_path, interpolate_rate, eyes_int_rate, return_instead=False, eye_batch_rate=1, device="cpu", disable_stats=False, save_output=False, file_process_fast=False, make_virtual_cam=False):
        self.disable_stats = disable_stats
        self.device = device
        self.eye_batch_rate = eye_batch_rate
        self.frame_counter = 0
        self.fps_poss_hist = deque(maxlen=self.avg_len)
        self.fps_hist = deque(maxlen=self.avg_len)
        self.pro_hist = deque(maxlen=self.avg_len)
        self.cpu_hist = deque(maxlen=self.avg_len)
        self.interpolate_rate = interpolate_rate
        self.eyes_int_rate = eyes_int_rate
        self.buffer = {}
        self.last_output_time = time.time()
        self.last_process_time = time.time()
        self.stop = False
        self.frame_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.return_instead_merker = []
        self.return_instead = return_instead
        self.stop = False
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        self.native_fps = cap.get(cv2.CAP_PROP_FPS)
        self.iswebcam = True if isinstance(video_path, int) else False
        self.interpolation_buffer = deque(maxlen=self.interpolate_rate)
        self.block_startframe = 0
        self.block_endframe = self.interpolate_rate+1
        self.is_first_frame = True
        self.save_output = save_output
        self.save_output_queue = multiprocessing.Queue()
        self.file_process_fast = file_process_fast

        self.proctime_get_face_rect = deque(maxlen=self.avg_len)
        self.proctime_get_face_cutout = deque(maxlen=self.avg_len)
        self.proctime_get_cutout_face_rect = deque(maxlen=self.avg_len)
        self.proctime_get_eyes_coords = deque(maxlen=self.avg_len)
        self.proctime_annotate_eyes = deque(maxlen=self.avg_len)

        self.make_virtual_cam = make_virtual_cam

        self.real_fps = deque(maxlen=self.avg_len)

        if self.save_output:
            os.makedirs("outputs", exist_ok=True)
        

        try:
            self.live_processing(cap)
        except KeyboardInterrupt:
            pass
        
        cap.release()
        cv2.destroyAllWindows()
        return np.array(self.return_instead_merker)



    

    def live_processing(self, cap):
        output_process = multiprocessing.Process(target=self.display_process, args=(self.frame_queue, self.native_fps, self.stop_event, self.interpolate_rate, self.disable_stats, self.save_output, self.save_output_queue, self.file_process_fast, self.make_virtual_cam))
        output_process.start()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                

            # process frame and add it to frame buffer
            self.buffer[self.frame_counter] = FrameElement(frame_id = self.frame_counter, frame = frame)
            self.process_frame(self.frame_counter)

            if (self.frame_counter) % (self.interpolate_rate+1) == 0 and self.frame_counter != 0:
                self.output_buffer_frames() # put ready frames into queue

            if self.stop_event.is_set():
                self.stop = True

            if self.stop:
                break

            self.frame_counter += 1

        # if save_output, save the collected frames to outputs
        if self.save_output:
            output_frames = []
          # print(f"saved about {self.save_output_queue.qsize()} frames")
            while True:
                # go through all frames until queue is empty
                try:
                    frame = self.save_output_queue.get(timeout=1)
                    output_frames.append(frame)
                except queue.Empty:
                    break
          # print(f"retrieved {len(output_frames)} frames")
            video_array = np.array(output_frames)
            out_file_name = self.video_path
            if "/" in out_file_name:
                out_file_name = out_file_name.split("/")[-1]
            if "\\" in out_file_name:
                out_file_name = out_file_name.split("/")[-1]
            if "." in out_file_name:
                out_file_name = out_file_name.split(".")[-2]
            misc.write_video(video_array, f"outputs/{out_file_name}.mp4", frame_rate = self.native_fps)


    
    # main part
    def process_frame(self, frame_id):    
        if self.iswebcam:
            self.buffer[frame_id].frame = cv2.flip(self.buffer[frame_id].frame, 1)

        # check if full annotation frame or interpolation frame
        self.buffer[frame_id].annotate_face = True if self.frame_counter % (self.interpolate_rate+1) == 0 else False
        self.buffer[frame_id].annotate_eyes = True if (self.buffer[frame_id].frame_id % (self.interpolate_rate+1) == 0 or self.buffer[frame_id].frame_id % (self.eyes_int_rate+1) == 0) else False
        
        self.get_face_rect(frame_id)



    def output_buffer_frames(self):
        if self.is_first_frame:
            loop_start = self.block_startframe
        else:
            loop_start = self.block_startframe+1
            
        for i in range(loop_start, self.block_endframe+1):
            self.get_face_cutout(i)

        if self.return_instead:
            for i in range(loop_start, self.block_endframe+1):
                if isinstance(self.buffer[i].cutout_frame, np.ndarray):
                    self.return_instead_merker.append(self.buffer[i].cutout_frame)
        else:
            self.get_eyes_coords()
                    
            for i in range(loop_start, self.block_endframe+1):
                self.annotate_eyes(i)
                try:
                    self.insert_eye_emojis(i) # für emojis auf den Augen
                    self.insert_histogram(i) # für histogram
                except:
                    pass

            if not self.disable_stats:
                for i in range(loop_start, self.block_endframe+1):
                    self.add_perf_info(i)
    
            for i in range(loop_start, self.block_endframe+1):
                self.frame_queue.put((self.buffer[i].frame, self.buffer[i].face_bbox, self.buffer[i].stats))

        self.is_first_frame = False

        # reset buffer, leave only last entry
        self.buffer = {self.block_endframe: self.buffer[self.block_endframe]}

        self.block_startframe = self.block_endframe
        self.block_endframe += self.interpolate_rate+1

    

    def display_process(self, frame_queue, native_fps, stop_event, interpolate_rate, disable_stats, save_output, save_output_queue, file_process_fast, make_virtual_cam):

        if not make_virtual_cam:
            last_output_time = time.time()
        
            while not stop_event.is_set():
                # print(frame_queue.qsize())
                try:
                    frame, annotations, stats = frame_queue.get(timeout=0.01)
                except queue.Empty:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break
                    continue
        
                # wait for next video frame time if processing too fast
                if (not self.iswebcam or frame_queue.qsize()<=interpolate_rate) and not file_process_fast:
                    while time.time() < last_output_time + (1/native_fps):
                        time.sleep(0.0001)
    
                self.real_fps.append(1/(time.time()-last_output_time))
                avg_real_fps = sum(self.real_fps)/len(self.real_fps)
                if not disable_stats:
                    cv2.putText(frame, f"output fps: {avg_real_fps:.1f} / {self.native_fps:.1f}", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                last_output_time = time.time()
                # time.sleep(0.2)
                
                # Display the annotated frame
                cv2.imshow('Annotated Video Feed', frame)
    
                if save_output:
                    save_output_queue.put(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

        else:
            last_output_time = time.time()
    
            # Create a virtual camera with the desired settings
            cam_width = 1280
            cam_height = 720
            cam_fps = native_fps
            cam_fmt = pyvirtualcam.camera.PixelFormat.BGR
    
            with pyvirtualcam.Camera(width=cam_width, height=cam_height, fmt=cam_fmt, fps=cam_fps, device='/dev/video4') as cam:
                while not stop_event.is_set():
                    try:
                        frame, annotations, stats = frame_queue.get(timeout=0.01)
                    except queue.Empty:
                        continue
    
                    # Resize the frame to match the virtual camera resolution
                    frame = cv2.resize(frame, (cam_width, cam_height))
    
                    # Wait for the next video frame time if processing too fast
                    if (not self.iswebcam or frame_queue.qsize() <= interpolate_rate) and not file_process_fast:
                        while time.time() < last_output_time + (1 / native_fps):
                            time.sleep(0.0001)
    
                    self.real_fps.append(1 / (time.time() - last_output_time))
                    avg_real_fps = sum(self.real_fps) / len(self.real_fps)
                    if not disable_stats:
                        cv2.putText(frame, f"output fps: {avg_real_fps:.1f} / {self.native_fps:.1f}", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    last_output_time = time.time()
    
                    # Send the frame to the virtual camera
                    # print("a")
                    cam.send(frame)
                    cam.sleep_until_next_frame()
    
                    if save_output:
                        save_output_queue.put(frame)



    

    

    def get_face_rect(self, frame_id):
        start_time = time.time()
        if self.buffer[frame_id].annotate_face:
            output = self.model_face.predict(self.buffer[frame_id].frame, conf=0.3, iou=0.5, imgsz=128, device=self.device, stream=False, half=False, mode="predict", verbose=False)
            output = [{'image_path': entry.path, 'faces': json.loads(entry.tojson())} for entry in output]
            if len(output[0]['faces']) > 0:
                self.buffer[frame_id].face_bbox = [int(output[0]['faces'][0]['box'][key]) for key in output[0]['faces'][0]['box']]
            else:
                self.buffer[frame_id].face_bbox = None
            
        self.buffer[frame_id].processing_times.append(time.time()-start_time)

    def get_cutout_face_rect(self, frame_id):
        start_time = time.time()
        face_bbox = self.buffer[frame_id].face_bbox
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            
            face_roi = self.buffer[frame_id].frame[int(y1+(y2-y1)*0.25):int(y1+(y2-y1)*0.55), x1:x2]
            self.buffer[frame_id].face_cutout_coords = [x1, int(y1+(y2-y1)*0.25), x2, int(y1+(y2-y1)*0.55)]
            desired_size = (640, 480)  # Set the desired size to match the original frame size
            self.buffer[frame_id].cutout_frame = cv2.resize(face_roi, desired_size)
            
        self.buffer[frame_id].processing_times.append(time.time()-start_time)

    def get_face_cutout(self, frame_id):
        start_time = time.time()
        if self.buffer[frame_id].face_bbox == None:
            face_rect_start = self.buffer[self.block_startframe].face_bbox
            face_rect_end = self.buffer[self.block_endframe].face_bbox
            face_rect = []
            if face_rect_start != None and face_rect_end != None:
                rate = (frame_id-self.block_startframe)/(self.block_endframe-self.block_startframe)
                for i in range(len(face_rect_start)):
                    face_rect.append(int((face_rect_start[i]*(1-rate)+face_rect_end[i]*rate)))
                self.buffer[frame_id].detected_face = True
                self.buffer[frame_id].face_bbox = face_rect
            else:
                self.buffer[frame_id].detected_face = False
                
            self.get_cutout_face_rect(frame_id)
        else:
            self.buffer[frame_id].detected_face = True
            self.get_cutout_face_rect(frame_id)

        self.buffer[frame_id].processing_times.append(time.time()-start_time)

    
            
    def annotate_eyes(self, frame_id):
        start_time = time.time()
        if self.buffer[frame_id].eyes_points == None:
            # interpolate
            eye_points_start = self.buffer[self.block_startframe].eyes_points
            eye_points_end = self.buffer[self.block_endframe].eyes_points
            eye_points_end_index = self.block_endframe
            for i in range(frame_id, self.block_endframe):
                if self.buffer[i].annotate_eyes:
                    eye_points_end = self.buffer[i].eyes_points
                    eye_points_end_index = i
            eye_coords = []
            if eye_points_start != None and eye_points_end != None:
                for i in range(len(eye_points_start)):
                    rate = (frame_id-self.block_startframe)/(eye_points_end_index-self.block_startframe)
                    eye_coords.append((int((eye_points_start[i][0]*(1-rate)+eye_points_end[i][0]*rate)), int((eye_points_start[i][1]*(1-rate)+eye_points_end[i][1]*rate))))
            self.buffer[frame_id].eyes_points = eye_coords

        # add points in frame
        comment = """
        for index, coord in enumerate(self.buffer[frame_id].eyes_points):
            coord = ((coord[0]), (coord[1]))
            coord = (int(coord[0]), int(coord[1]))
            cv2.circle(self.buffer[frame_id].frame, coord, 2, (0, 255, 0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            fontColor = (255,255,255)
            thickness = 1
            lineType = 2
            cv2.putText(self.buffer[frame_id].frame, str(index), coord, font, fontScale, fontColor, thickness, lineType)
            """

        self.buffer[frame_id].processing_times.append(time.time()-start_time)

    def insert_eye_emojis(self, frame_id):

        y_scaling = 1.2 # 2.0 would be normal
        x_scaling = 1.6 # 2.0 would be normal
        
        if len(self.buffer[frame_id].eyes_points) == 10:
            right_outer = self.buffer[frame_id].eyes_points[0]
            right_middle = self.buffer[frame_id].eyes_points[1]
            right_inner = self.buffer[frame_id].eyes_points[2]
            right_upper = self.buffer[frame_id].eyes_points[6]
            right_lower = self.buffer[frame_id].eyes_points[7]
        
            right_width = abs(math.ceil(right_outer[0]) - math.ceil(right_inner[0]))
            right_height = abs(math.ceil(right_upper[1]) - math.ceil(right_lower[1]))
        
          # print(right_width, right_height)
            if right_width > 1 and right_height > 1:  # Adjust the threshold as needed

                if right_height > (right_width/4):
                    image_to_use = image_eye_open
                else:
                    image_to_use = image_eye_closed
                
        
                
                
                eye_roi = self.buffer[frame_id].frame[math.floor(right_middle[1] - right_height / y_scaling): math.ceil(right_middle[1] + right_height / y_scaling),
                                                      math.ceil(right_middle[0] - right_width / x_scaling): math.ceil(right_middle[0] + right_width / x_scaling)]
                resized_emoji = cv2.resize(image_to_use[:, :, :3], (eye_roi.shape[1], eye_roi.shape[0]))
                resized_alpha = cv2.resize(image_to_use[:, :, 3], (eye_roi.shape[1], eye_roi.shape[0]))
                mask = resized_alpha / 255.0
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
              # print("mask.shape:", mask.shape)
              # print("resized_alpha.shape:", resized_alpha.shape)
              # print("eye_roi.shape:", eye_roi.shape)
              # print("right middle[1]:", right_middle[1])
              # print((right_middle[0] - right_width // 2))
              # print((right_middle[0] + right_width // 2))
                blended_roi = (eye_roi * (1 - mask) + resized_emoji * mask).astype(np.uint8)
                self.buffer[frame_id].frame[math.floor(right_middle[1] - right_height / y_scaling): math.ceil(right_middle[1] + right_height / y_scaling),
                                                      math.ceil(right_middle[0] - right_width / x_scaling): math.ceil(right_middle[0] + right_width / x_scaling)] = blended_roi
            
                    
            left_outer = self.buffer[frame_id].eyes_points[5]
            left_middle = self.buffer[frame_id].eyes_points[4]
            left_inner = self.buffer[frame_id].eyes_points[3]
            left_upper = self.buffer[frame_id].eyes_points[8]
            left_lower = self.buffer[frame_id].eyes_points[9]

            left_width = abs(math.ceil(left_outer[0]) - math.ceil(left_inner[0]))
            left_height = abs(math.ceil(left_upper[1]) - math.ceil(left_lower[1]))
        
          # print(left_width, left_height)
            if left_width > 1 and left_height > 1:  # Adjust the threshold as needed

                if left_height > (left_width/4):
                    image_to_use = image_eye_open
                else:
                    image_to_use = image_eye_closed
                
        
                
                eye_roi = self.buffer[frame_id].frame[math.floor(left_middle[1] - left_height / y_scaling): math.ceil(left_middle[1] + left_height / y_scaling),
                                                      math.ceil(left_middle[0] - left_width / x_scaling): math.ceil(left_middle[0] + left_width / x_scaling)]
                resized_emoji = cv2.resize(image_to_use[:, :, :3], (eye_roi.shape[1], eye_roi.shape[0]))
                resized_alpha = cv2.resize(image_to_use[:, :, 3], (eye_roi.shape[1], eye_roi.shape[0]))
                mask = resized_alpha / 255.0
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
              # print("mask.shape:", mask.shape)
              # print("resized_alpha.shape:", resized_alpha.shape)
              # print("eye_roi.shape:", eye_roi.shape)
              # print("left middle[1]:", left_middle[1])
              # print((left_middle[0] - left_width // 2))
              # print((left_middle[0] + left_width // 2))
                blended_roi = (eye_roi * (1 - mask) + resized_emoji * mask).astype(np.uint8)
                self.buffer[frame_id].frame[math.floor(left_middle[1] - left_height / y_scaling): math.ceil(left_middle[1] + left_height / y_scaling),
                                                      math.ceil(left_middle[0] - left_width / x_scaling): math.ceil(left_middle[0] + left_width / x_scaling)] = blended_roi


    

    def insert_histogram(self, frame_id):
        frame = self.buffer[frame_id].frame
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR colors for histogram lines
        hist_img = np.zeros((1000, 256, 3), dtype=np.uint8)
    
        # Calculate and draw the original histogram
        for i, color in enumerate(colors):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX)
    
            for j in range(1, 256):
                cv2.line(hist_img, (j - 1, 200 - int(hist[j - 1])),
                         (j, 200 - int(hist[j])), color, 1)
    
        # Calculate and draw the equalized histogram
        eq_frame = np.copy(frame)
        for i in range(3):
            eq_frame[:, :, i] = cv2.equalizeHist(eq_frame[:, :, i])
    
        for i, color in enumerate(colors):
            eq_hist = cv2.calcHist([eq_frame], [i], None, [256], [0, 256])
            cv2.normalize(eq_hist, eq_hist, 0, 200, cv2.NORM_MINMAX)
    
            for j in range(1, 256):
                cv2.line(hist_img, (j - 1, 400 - int(eq_hist[j - 1])),
                         (j, 400 - int(eq_hist[j])), color, 1)
    
        # Edge detection
        edge_frame = cv2.Canny(frame, 100, 200) # kann auch mit mehreren kerneln np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) usw gemacht werden
        edge_frame = cv2.cvtColor(edge_frame, cv2.COLOR_GRAY2BGR)
        edge_frame = cv2.resize(edge_frame, (200, 200))
        hist_img[400:600, 28:228] = edge_frame
    
        # Simple blurring
        blur_frame = cv2.GaussianBlur(frame, (5, 5), 0) # kann auch mit kernel np.array([[0.5, 1, 0.5], [1, 3, 1], [0.5, 1, 0.5]]) gemacht werden
        blur_frame = cv2.resize(blur_frame, (200, 200))
        hist_img[600:800, 28:228] = blur_frame
    
        # Simple sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp_frame = cv2.filter2D(frame, -1, kernel)
        sharp_frame = cv2.resize(sharp_frame, (200, 200))
        hist_img[800:1000, 28:228] = sharp_frame
    
        hist_img = cv2.resize(hist_img, (200, 400))
        self.buffer[frame_id].frame[10:410, self.buffer[frame_id].frame.shape[1] - 210:self.buffer[frame_id].frame.shape[1] - 10] = hist_img

        

    def get_eyes_coords(self):
        start_time = time.time()

        # detect eye points for all not interpolate frames
        do_detect_extra = False
        do_detect_less = False
        num_do_detect_less = 0
        image_size = self.eye_model_size if self.eye_model_size != None else 512
        detect_frames = [self.buffer[frame_id].cutout_frame for frame_id in self.buffer if self.buffer[frame_id].annotate_eyes and self.buffer[frame_id].eyes_points == None and isinstance(self.buffer[frame_id].cutout_frame, np.ndarray)]
        # print("len detect frames:", len(detect_frames))
        
        if len(detect_frames) > self.eye_batch_rate:
            do_detect_extra = True
            detect_extra = [detect_frames.pop(0)] * self.eye_batch_rate
            output_extra = self.model_eyes.predict(detect_extra, conf=0.3, iou=0.5, imgsz=image_size, device=self.device, stream=False, half=False, mode="predict", verbose=False)
            output_extra = [{'image_path': entry.path, 'eyes': json.loads(entry.tojson())} for entry in output_extra[0:1]]

        if len(detect_frames) > 0:

            if len(detect_frames) < self.eye_batch_rate:
                do_detect_less = True
                num_do_detect_less = self.eye_batch_rate-len(detect_frames)
                detect_frames = detect_frames + [detect_frames[-1]]*(num_do_detect_less)

            output = self.model_eyes.predict(detect_frames, conf=0.3, iou=0.5, imgsz=image_size, device="cpu", stream=False, half=False, mode="predict", verbose=False)
            output = [{'image_path': entry.path, 'eyes': json.loads(entry.tojson())} for entry in output]

            if do_detect_less:
                output = output[:-do_detect_less]
    
            if do_detect_extra:
                output = output + output_extra
    
            
    
            # add the points to the respective FrameElement
            eye_coords_merker = []
            for out in output:
                if len(out['eyes']) > 0:
                    eye_coords = []
                    for i in range(len(out['eyes'][0]['keypoints']['x'])):
                        eye_coords.append((out['eyes'][0]['keypoints']['x'][i], out['eyes'][0]['keypoints']['y'][i]))
                else:
                    eye_coords = None
                eye_coords_merker.append(eye_coords)
    
            # for index, entry in enumerate(eye_coords_merker):
            #   # print(f"eye_coords_merker[{index}]:", len(entry))
    
            for frame_id in self.buffer:
                if self.buffer[frame_id].annotate_eyes and self.buffer[frame_id].eyes_points == None and isinstance(self.buffer[frame_id].cutout_frame, np.ndarray):
                    self.buffer[frame_id].eyes_points = eye_coords_merker.pop(0)
    
            
    
            # do interpolation
            for frame_id in self.buffer:
                if self.buffer[frame_id].eyes_points != None and (frame_id != self.block_startframe or frame_id == 0):
                    eye_coords = self.buffer[frame_id].eyes_points
                    height, width, _ = self.buffer[frame_id].cutout_frame.shape
                    for index_keypoint, coord in enumerate(eye_coords):
                        cutout_frame_width, cutout_frame_height = self.buffer[frame_id].face_cutout_coords[2]-self.buffer[frame_id].face_cutout_coords[0], self.buffer[frame_id].face_cutout_coords[3]-self.buffer[frame_id].face_cutout_coords[1]
                        eye_coords[index_keypoint] = ((coord[0]/width)*cutout_frame_width+self.buffer[frame_id].face_cutout_coords[0], (coord[1]/height)*cutout_frame_height+self.buffer[frame_id].face_cutout_coords[1])
                    self.buffer[frame_id].eyes_points = eye_coords

        # add info about processing time (time per frame is avg of block processing time / block size)
        # if it's the first block, include start frame (0) in calculation
        for frame_id in self.buffer:
            if frame_id != self.block_startframe or frame_id == 0:
                if self.block_startframe == 0:
                    self.buffer[frame_id].processing_times.append((time.time()-start_time)/(self.block_endframe-self.block_startframe+1))
                else:
                    self.buffer[frame_id].processing_times.append((time.time()-start_time)/(self.block_endframe-self.block_startframe))

    
    def add_perf_info(self, frame_id):
        processing_time = sum(self.buffer[frame_id].processing_times)
        self.pro_hist.append(processing_time)

        self.proctime_get_face_rect.append(self.buffer[frame_id].processing_times[0])
        self.proctime_get_face_cutout.append(self.buffer[frame_id].processing_times[1])
        self.proctime_get_cutout_face_rect.append(self.buffer[frame_id].processing_times[2])
        self.proctime_get_eyes_coords.append(self.buffer[frame_id].processing_times[3])
        self.proctime_annotate_eyes.append(self.buffer[frame_id].processing_times[4])

        avg_processing_time = (sum(self.pro_hist)/len(self.pro_hist))
        avg_proctime_get_face_rect = (sum(self.proctime_get_face_rect)/len(self.proctime_get_face_rect))
        avg_proctime_get_face_cutout = (sum(self.proctime_get_face_cutout)/len(self.proctime_get_face_cutout))
        avg_proctime_get_cutout_face_rect = (sum(self.proctime_get_cutout_face_rect)/len(self.proctime_get_cutout_face_rect))
        avg_proctime_get_eyes_coords = (sum(self.proctime_get_eyes_coords)/len(self.proctime_get_eyes_coords))
        avg_proctime_annotate_eyes = (sum(self.proctime_annotate_eyes)/len(self.proctime_annotate_eyes))

        # Display FPS and processing time on the frame
        # cv2.putText(frame, f"FPS:  {avg_fps} / {self.native_fps} ({avg_poss_fps})", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(self.buffer[frame_id].frame, f"{(avg_processing_time*1000):.1f} ms", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(self.buffer[frame_id].frame, f"{(1/avg_processing_time):.1f} fps", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(self.buffer[frame_id].frame, f"{(avg_proctime_get_face_rect*1000):.1f} ms (get_face_rect)", (5, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(self.buffer[frame_id].frame, f"{(avg_proctime_get_face_cutout*1000):.1f} ms (get_face_cutout)", (5, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(self.buffer[frame_id].frame, f"{(avg_proctime_get_cutout_face_rect*1000):.1f} ms (get_cutout_face_rect)", (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(self.buffer[frame_id].frame, f"{(avg_proctime_get_eyes_coords*1000):.1f} ms (get_eyes_coords)", (5, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(self.buffer[frame_id].frame, f"{(avg_proctime_annotate_eyes*1000):.1f} ms (annotate_eyes)", (5, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # cv2.putText(frame, f"CPU:  {avg_cpu} %", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if self.buffer[frame_id].detected_face == False:
            cv2.putText(self.buffer[frame_id].frame, f"no face detected", (5, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    
    def reduce_frame(self, frame, size):
        # Get the original dimensions of the frame
        height, width, _ = frame.shape
    
        # Calculate the aspect ratio of the frame
        aspect_ratio = width / height
    
        # Scale the frame to a height of 480 while maintaining the aspect ratio
        new_height = size
        new_width = int(new_height * aspect_ratio)
        resized_frame = cv2.resize(frame, (new_width, new_height))
    
        # Calculate the padding dimensions to make the frame square
        square_size = max(new_height, new_width)
        pad_height = (square_size - new_height) // 2
        pad_width = (square_size - new_width) // 2
    
        # Create a new square frame with black padding
        padded_frame = np.zeros((square_size, square_size, 3), dtype=np.uint8)
        padded_frame[pad_height:pad_height+new_height, pad_width:pad_width+new_width] = resized_frame
    
        # Scale the height of the padded frame back to 480 if it was changed by the padding
        if square_size != size:
            final_frame = cv2.resize(padded_frame, (size, size))
        else:
            final_frame = padded_frame
    
        return final_frame