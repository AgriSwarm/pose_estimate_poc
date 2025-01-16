import cv2
import json
import tempfile
import numpy as np
from tqdm import tqdm
from pathlib import Path
from detector.predict_shot import predict
from transform import transform_world_to_camera

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def compute_rotation_matrix(vec):
    x_axis = normalize(vec)
    y_axis = normalize(np.cross([0, 0, 1], x_axis))  # Assuming Z is up
    z_axis = np.cross(x_axis, y_axis)
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    return rotation_matrix

def main():
    dataset_path = Path("./data")
    dataset_path.mkdir(exist_ok=True)

    transformed = transform_world_to_camera()
    transforms_results = {}
    pbar = tqdm(transformed)
    pbar.set_description("Create dataset")
    for data in pbar:
        name = data["name"]
        image = data["image"]
        centroids = np.array(data["centroids"])
        vec = centroids[0] - centroids[1]
        rotation_matrix = compute_rotation_matrix(vec)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            cv2.imwrite(temp_file.name, image)
            temp_image_path = temp_file.name

            with open(temp_image_path, 'rb') as file:
                image_binary = file.read()
                results = predict(image_binary)

        results = sorted(results, key = lambda x: x["bbox_xyxy"][0])
        if len(results) == 0:
            cropped_image = image
        else:
            x1,y1,x2,y2 = [int(z) for z in results[0]["bbox_xyxy"]]
            cropped_image = image[y1:y2, x1:x2]

        filename = f"{name}.jpg"
        transforms_results[filename] = {
            "R": rotation_matrix.tolist()
        }
        filename = dataset_path / filename
        cv2.imwrite(f"{filename}", cropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    json.dump(transforms_results, open("./data/datasets.json", "w"))


if __name__ == "__main__":
    main()