from .imports import *

class RandomObstruction(torch.nn.Module):
    def __init__(self, p=0.5, max_size=0.3):
        super().__init__()
        self.p = p
        self.max_size = max_size

    def forward(self, img):
        if random.random() < self.p:
            _, h, w = img.shape
            size = random.uniform(0.1, self.max_size)
            aspect_ratio = random.uniform(0.5, 2.0)
            obstruction_w = int(w * size * aspect_ratio)
            obstruction_h = int(h * size / aspect_ratio)
            x = random.randint(0, w - obstruction_w)
            y = random.randint(0, h - obstruction_h)
            img[:, y:y+obstruction_h, x:x+obstruction_w] = 0
        return img

def train_model(pretrained_model='models/yolov8n-pose.pt', save_name="default_name", imgsz=64, epochs=200, batch=64, workspace, usecomet=True):
    if usecomet:
        experiment = Experiment(api_key=api_key, project_name="yolov8finetune_eyes", workspace=workspace)

    
    
    try:
        
        # Load a model
        model = YOLO(pretrained_model)  # load a pretrained model (recommended for training)
        #model = YOLO('train_sizeswitch_holder/last_1024_2.pt')
        #model = YOLO('yolov8m-pose.yaml')
        #model.export(format='engine')
        #model = YOLO('yolov8m-pose.engine')
        
        # Train the model
        results = model.train(
            model=model,
            data='data/yolo_pose_dataset.yaml',
            save=True,
            patience=0,
            val=True,
            device=0,
            workers=8,
            verbose=False,
            pretrained=True,
            plots=True,
            box=7, cls=0.5, dfl=1, pose=12, kobj=2,
            cache=True,
            name=save_name, # Same as your comet project name
            project='yolov8finetune_eyes', # Same as your comet project name
            
            epochs=epochs,
            
            imgsz=imgsz, 
            
            batch=batch, 
            
            lr0=1e-3, 
            lrf=1e-2, 
            warmup_epochs=10,
            
            optimizer="AdamW", # SGD, Adam, AdamW, NAdam, RAdam, RMSProp
            exist_ok=True,
            freeze=None,
            deterministic=False,
            
            resume=False,
    
            hsv_h=0.4, # default 0.015
            hsv_s=1.0, # default 0.7
            hsv_v=0.7, # default 0.4
            # scale=0.8,
    
            mosaic=0.0,
            degrees=30,
            shear=20,
            perspective=0.000,
            
            #mixup=0.05,
            #bgr=0.05,
            #fliplr=0.05,
            #flipud=0.05,
            #crop_fraction=1.0,
            crop_fraction=0.1,
            copy_paste=0.2,

        )
        if usecomet:
            experiment.end()
    except Exception as e:
        print(e)
        if usecomet:
            experiment.end()

    # copy resulting model to models folder
    shutil.copyfile(f"yolov8finetune_eyes/{save_name}/weights/last.pt", f"models/eyes_{imgsz}_auto.pt")

def init_comet(api_key=""):
    comet_ml.init(api_key=api_key)

def convert_pt_to_onnx(model_path='models/eyes_best_128_v2.pt', imgsz=128, batch=1, form="onnx"):
    model = YOLO(model_path)
    model.export(format=form, half=True, simplify=True, imgsz=imgsz, batch=batch)

def multiply_training_data(rate):
    if rate <= 1:
        return False
    train_files = glob.glob("data/03_train_split/**.txt")
    train_files = [file.split("/")[-1].split(".")[0] for file in train_files]

    for i in range(rate-1):
        for train_file in train_files:
            copyfile(f"data/03_train_split/{train_file}.txt", f"data/03_train_split/{train_file}_{i}.txt")
            copyfile(f"data/03_train_split/{train_file}.PNG", f"data/03_train_split/{train_file}_{i}.PNG")
    

def prepare_video_annotations_for_training(val_ratio=0.1, multiply_train_data=1):
    # make sure folders are clean
    for folder in ["data/02_annotated_frames"]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
                
    video_annotation_to_image_annotations(
        annotation_paths = glob.glob("data/01_annotations_raw/**.json"),
        folder_out = "data/02_annotated_frames"
    )
    merged_annotations = merge_json_files(glob.glob("data/02_annotated_frames/**.json"))
    bundled_annotations = merged_data_bundle_images(merged_annotations)
    coco_to_yolo(bundled_annotations, "data/02_annotated_frames")
    make_train_val_splits("data/02_annotated_frames", val_ratio)

    multiply_training_data(rate=multiply_train_data)

def make_train_val_splits(all_images_folder, val_ratio):
    os.makedirs("data/03_train_split", exist_ok=True)
    os.makedirs("data/03_val_split", exist_ok=True)

    # make sure folders are clean
    for folder in ["data/03_train_split", "data/03_val_split"]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    all_files = glob.glob(f"{all_images_folder}/**.txt")
    all_files = [file.split("/")[-1].split(".")[0] for file in all_files]
    files_train, files_val = split_list(all_files, val_ratio)
    for file_train in files_train:
        copyfile(f"{all_images_folder}/{file_train}.txt", f"data/03_train_split/{file_train}.txt")
        copyfile(f"{all_images_folder}/{file_train}.PNG", f"data/03_train_split/{file_train}.PNG")
    for file_val in files_val:
        copyfile(f"{all_images_folder}/{file_val}.txt", f"data/03_val_split/{file_val}.txt")
        copyfile(f"{all_images_folder}/{file_val}.PNG", f"data/03_val_split/{file_val}.PNG")

def split_list(string_list, ratio):
    val_list_size = int(len(string_list) * ratio)
    val_list = random.sample(string_list, val_list_size)
    train_list = [item for item in string_list if item not in val_list]

    return train_list, val_list

def write_video(video_array, out_path):
    output_file = out_path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = video_array.shape[1:]
    video_writer = cv2.VideoWriter(output_file, fourcc, 30, (width, height))
    
    for frame in video_array:
        frame_bgr = frame # cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    video_writer.release()

def process_video(video_path, processor, interpolate_rate):
    out = processor.start(video_path=video_path, interpolate_rate=interpolate_rate, return_instead=True)
    output_file = f'outputs/{"_".join(video_path.split("/")[-2:])}_output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if out.size == 0:
        return None
    
    height, width, _ = out.shape[1:]
    video_writer = cv2.VideoWriter(output_file, fourcc, 30, (width, height))
    
    for frame in out:
        frame_bgr = frame # cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    video_writer.release()

def scale_images(image_paths):
    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size

            if height > 720:
                new_height = 720
                new_width = int((new_height / height) * width)
                img = img.resize((new_width, new_height))

            base, ext = os.path.splitext(path)
            new_filename = f"{base}_scaled.jpg"

            img.save(new_filename, 'JPEG')

            print(f"Saved scaled image as {new_filename}")

def video_annotation_to_image_annotations(annotation_paths, folder_out):
    if not isinstance(annotation_paths, list):
        annotation_paths = [annotation_paths]

    for index, annotation_path in enumerate(annotation_paths):
        with open(annotation_path) as f:
            data = json.load(f)
        
        os.makedirs(folder_out, exist_ok=True)
    
        annotated_images = [annotation['image_id'] for annotation in data['annotations']]
        data['images'] = [image for image in data['images'] if image['id'] in annotated_images]

        video_path = ".".join(annotation_path.split(".")[:-1]+["mp4"])
        cap = cv2.VideoCapture(video_path)
        
        # Extract frames and update the JSON file
        for image in data['images']:
            frame_name = image['file_name']
            frame_path = os.path.join(folder_out, f"{index}_"+frame_name)
            
            frame_number = int(frame_name.split('_')[1].split('.')[0])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read() # Read the frame from the video
            
            if ret:
                cv2.imwrite(frame_path, frame)
                image['file_name'] = f"{index}_"+frame_name
            else:
                print(f"Failed to extract frame {frame_number}")
        
        cap.release()
        
        # Save the updated JSON file
        with open(f'{folder_out}/updated_annotations_{index}.json', 'w') as f:
            json.dump(data, f, indent=4)

def cocokeypoints_to_yolo(bundled_data, output_dir, merge_face_eyebrows=False):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for image_annotations_key in bundled_data['annotations']:
        image_info = bundled_data['images'][image_annotations_key]
        annotations = [annotation for annotation in bundled_data['annotations'][image_annotations_key]]
        
        """
        for keypoint in cvat_data['categories']:
            if keypoint['id'] not in classes_merker:
                classes_merker.append(keypoint['id'])
            if len(keypoint['keypoints'])*3 > max_keypoint_len:
                max_keypoint_len = len(keypoint['keypoints'])*3
        """

        annotations_merged = []

        # make face-eyebrow pairs
        pairs = []
        face_centers = [annotation['annotation_bbox_center'] for annotation in annotations if annotation['annotation_class'] == 0]
        
        for index, annotation in enumerate(annotations):
            if annotation['annotation_class'] == 1:  # eyebrow
                # get closest face
                closest_face_center = min(face_centers, key=lambda x: abs(x - annotation['annotation_bbox_center']))
                
                for subindex, subannotation in enumerate(annotations):
                    if subannotation['annotation_class'] == 0 and subannotation['annotation_bbox_center'] == closest_face_center:
                        pairs.append((subindex, index))
                        break
        
        # print(pairs, image_info['file_name'])
        pairs = [[pair[0] for pair in pairs], [pair[1] for pair in pairs]] # faces, eyebrows
                

        if not merge_face_eyebrows:
            # Iterate over each image and its annotations
            for index, annotation in enumerate(annotations):
                if annotation['annotation_class'] == 1:
                    continue
                image_name = image_info['file_name']
                image_width = image_info['width']
                image_height = image_info['height']
        
                # Create the YOLO annotation file path
                yolo_annotation_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
                
                class_index = annotation['annotation_class']
                bbox = annotation['annotation_bbox']
                keypoints = annotation['annotation_keypoints']
                
                # Normalize the bounding box coordinates
                x_center = (bbox[0] + bbox[2] / 2) / image_width
                y_center = (bbox[1] + bbox[3] / 2) / image_height
                width = bbox[2] / image_width
                height = bbox[3] / image_height
        
                # Write the YOLO annotation line
                yolo_line = f"{0} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        
                # Normalize and add the keypoint coordinates to the YOLO line
                for i in range(0, len(keypoints), 3):
                    x = keypoints[i] / image_width
                    y = keypoints[i + 1] / image_height
                    visibility = keypoints[i + 2]
                    yolo_line += f" {x:.6f} {y:.6f} {visibility}"

                annotations_merged.append(yolo_line)

        else:
            # go through faces only and add pair eyebrows
            for index, annotation in enumerate(annotations):
                if annotation['annotation_class'] == 1:
                    continue
                image_name = image_info['file_name']
                image_width = image_info['width']
                image_height = image_info['height']
        
                # Create the YOLO annotation file path
                yolo_annotation_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
                
                class_index = annotation['annotation_class']
                bbox = annotation['annotation_bbox']
                keypoints = annotation['annotation_keypoints']
                
                # Normalize the bounding box coordinates
                x_center = (bbox[0] + bbox[2] / 2) / image_width
                y_center = (bbox[1] + bbox[3] / 2) / image_height
                width = bbox[2] / image_width
                height = bbox[3] / image_height
        
                # Write the YOLO annotation line
                yolo_line = f"{0} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        
                # Normalize and add the keypoint coordinates to the YOLO line
                for i in range(0, len(keypoints), 3):
                    x = keypoints[i] / image_width
                    y = keypoints[i + 1] / image_height
                    visibility = keypoints[i + 2]
                    if visibility == 1:
                        visibility = 0
                    yolo_line += f" {x:.6f} {y:.6f} {visibility}"
                if index in pairs[0]:
                    paired_eyebrows_index = pairs[1][pairs[0].index(index)]
                    paired_eyebrows = annotations[paired_eyebrows_index]

                    pair_keypoints = paired_eyebrows['annotation_keypoints']
                    for i in range(0, len(pair_keypoints), 3):
                        x = pair_keypoints[i] / image_width
                        y = pair_keypoints[i + 1] / image_height
                        visibility = pair_keypoints[i + 2]
                        yolo_line += f" {x:.6f} {y:.6f} {visibility}"
                else:
                    for i in range(0, 18, 3):
                        yolo_line += f" {0.0} {0.0} {0}"
    
                
    
                annotations_merged.append(yolo_line)
    
        with open(yolo_annotation_path, 'w') as f:
            f.write("\n".join(annotations_merged))

    print("Conversion completed.")

def coco_to_yolo(annotation_file, output_folder):
    cocokeypoints_to_yolo(annotation_file, output_folder, merge_face_eyebrows=False)

def merge_json_files(filepaths):
    merged_data = {
        "licenses": [],
        "info": {},
        "categories": [],
        "images": [],
        "annotations": []
    }

    image_id_map = {}
    annotation_id = 1
    image_id = 1

    for filepath in filepaths:
        with open(filepath, 'r') as file:
            data = json.load(file)

            # Merge licenses
            merged_data["licenses"].extend(data["licenses"])

            # Merge info
            merged_data["info"].update(data["info"])

            # Merge categories
            merged_data["categories"].extend(data["categories"])

            # Merge images
            for image in data["images"]:
                old_image_id = image["id"]
                image["id"] = image_id
                image_id_map[old_image_id] = image_id
                merged_data["images"].append(image)
                image_id += 1

            # Merge annotations
            for annotation in data["annotations"]:
                annotation["id"] = annotation_id
                annotation["image_id"] = image_id_map[annotation["image_id"]]
                merged_data["annotations"].append(annotation)
                annotation_id += 1

    return merged_data

def merged_data_bundle_images(merged_data):
    bundled_data = {}
    bundled_data['info'] = merged_data['info']
    bundled_data['categories'] = []
    bundled_data['images'] = {}
    for i in range(len(merged_data['images'])):
        bundled_data['images'][i+1] = merged_data['images'][i]
    bundled_data['annotations'] = {}

    categories = {}
    num_categories = 0

    for category in merged_data['categories']:
        if category['id'] not in categories:
            categories[category['id']] = num_categories
            new_category = category
            new_category['id'] = num_categories
            bundled_data['categories'].append(category)
            num_categories += 1
    
    for annotation in merged_data['annotations']:
        
        related_image = annotation['image_id']

        append_dict = {}
        append_dict['image_id'] = annotation['image_id']
        append_dict['annotation_id'] = annotation['id']
        append_dict['annotation_class'] = categories[annotation['category_id']]
        append_dict['annotation_bbox'] = annotation['bbox']
        append_dict['annotation_bbox_center'] = np.mean(annotation['bbox'])
        append_dict['annotation_keypoints'] = annotation['keypoints']
        
        if related_image in bundled_data['annotations']:
            bundled_data['annotations'][related_image].append(append_dict)
        else:
            bundled_data['annotations'][related_image] = [append_dict]

    return bundled_data