import cv2
import json
import numpy as np
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
    results = {}
    for data in transformed:
        name = data["name"]
        image = data["image"]
        centroids = np.array(data["centroids"])
        vec = centroids[0] - centroids[1]
        rotation_matrix = compute_rotation_matrix(vec)

        results = predict(image)
        results = sorted(results, lambda x: x["bbox_xyxy"][0])
        bbox = results[0]
        cropped_image = cv2.crop(image, bbox)

        filename = f"{name}.jpg"
        cropped_image.save(filename)
        results[filename].append({
            "R": rotation_matrix
        })
        cropped_image.save(dataset_path / filename)
    json.dump(results, open("./data/datasets.json", "w"))


if __name__ == "__main__":
    main()