### Eye Tracking with yolov8

uses [YOLOv8n-Face](https://github.com/lindevs/yolov8-face) for face recognition.  
if face gets recognized, uses my finetuned model, which is based on the pretrained [YOLOv8n-pose](https://docs.ultralytics.com/tasks/pose/) by ultralytics.  
it is optimized for low resource cpu usage and can handle (depending on settings) 120fps live video (like webcam).  
can also be used on video files.  
for webcam use, set input_source to 0 in the notebook.  

[![Watch the video](outputs/youtube_vid_01.mp4)]  
https://youtu.be/bHD4gBP1GRo  
  
  
[![Watch the video](outputs/youtube_vid_02.mp4)]  
https://youtu.be/P1vwnUpfsg4  
  
  
[![Watch the video](outputs/youtube_vid_03.mp4)]  
https://youtu.be/DF364juVpGo  
  
  
[![Watch the video](outputs/youtube_vid_04.mp4)]  
https://youtu.be/OuELBU91EFA  
  
  

### Training of eye recognition model
1. Put training video in example_video/.
2. Crop it by running video processor with the video file as source with argument return_instead=True. This will return cropped faces.
3. Save the cropped face video with write_video(output, path_to_save)
4. Annotate the cropped video (i used CVAT).
5. Export annotations in coco-keypoints format.
6. Put the cropped video and the annotations file in data/01_annotations_raw/. Make sure both files have the same name, with the video file ending in .mp4 and the annotation file ending in .json.
7. run prepare_video_annotations_for_training(val_ratio=0.1, multiply_train_data=16) in the notebook. this will automatically convert the annotations to the correct format, extract the relevant video frames and split the data into test and train/validation. multiply_train_data will duplicate the train data, very useful for training on high end system with small amount of data since it allows the preprocessing and training of each epoch to be more efficient (since there is always a delay when a new epoch starts, you want to stay in one epoch for more steps, preferably without having to reduce batch size. Duplicates are not a worry here since the data augmentor of yolov8.train augments the images pretty well).
8. run the training cell. By default it will train models for image input sizes 64, 128, 256, 512 respectively. You can only train only one model as well. the resulting model gets automatically placed into models/.  

### Using it
1. Set the input source (0 for webcam or a path to a video file).
2. modify interpolate_rate, eyes_int_rate and eye_model_size. setting eyes_int_rate (eyes interpolation rate) higher than interpolate_rate (face detection interpolation rate) will result in the same effect as setting eyes_int_rate to interpolate_rate. higher interpolate rate will be more performant, but might result in inaccurate annotations if too high. eye_model_size sets the image size for the eye model. most be a value that you trained a model for (in step 8 for training).
3. when you run the cell, the model will be transformed to onnx format (faster on cpu) with a batch size suitable for your interpolation settings.  

The VideoProcessor will now try to detect a face on the first frame, do nothing for interpolate_rate frames and then try to detect a face again. if a face has been detected both times, it interpolates all frames in between.  
Then the first frame, the last frame and all frames in between with step size of eyes_int_rate+1 will try to detect eyes. if eyes have been detected at least two times, the frames in between (if there are frames in between) will have their eye annotations interpolated.  
The image that the eye detection model uses as input it a cropped out part of the recognized face of each frame (the middle-top part where the eyes are expected to be). Because it only uses this small crop of the frame, it allows the eye model to work with much smaller images making it very efficient.  
The eye model also uses batches, as mentioned earlier, to detect the eyes on the not-interpolate frames all at once, further improving performance.  
The higher interpolate_rate is and the lower eyes_int_rate is, the larger the batch will be, improving overall efficiency.  

However, increasing interpolate_rate will increate the video output delay. If interpolate_rate is too large and the head in the video is moves too fast, it may result in the eyes not being within the interpolated head_bbox anymore, resulting in no eye detection.
