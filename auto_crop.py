from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import cv2

def check_valid_image(path):
    img = cv2.imread(str(path))
    return img is not None

def get_valid_images():
    images = list(Path("images").glob("*"))
    image_paths = [img for img in images if check_valid_image(img)]
    return image_paths
    
print("[load] YOLO v8 ...", end="", flush=True)
model = YOLO('yolov8x.pt')
print("fin")

image_paths = get_valid_images()
batch_size = 10
for i in tqdm(range(len(image_paths)//batch_size)):
    minibatch_img_path = image_paths[i*batch_size:(i+1)*batch_size]
    results = model.predict(minibatch_img_path, save=True, conf=0.1, save_dir="results")  
    for img_path, result in zip(minibatch_img_path,results):
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        class_names = result.names
        for box in boxes:
            cls_idx = int(box.cls.detach().cpu().numpy())
            pred_label = class_names[cls_idx]
            pred_bbox = box.xyxy
            print(img_path)
            print(pred_label)
            print(pred_bbox)