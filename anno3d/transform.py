import math
import glob
import json
import cv2
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from tqdm import tqdm

DEBUG = False

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def transform_vector(vec, transform_matrix):
    vec_homogeneous = np.append(vec, 1)
    transformed_vec_homogeneous = transform_matrix @ vec_homogeneous
    return transformed_vec_homogeneous[:3] / transformed_vec_homogeneous[3]

def project_to_image_plane(point, intrinsic_matrix, dist_coeffs):
    point = point.reshape(1, 1, -1)
    point_2d = cv2.projectPoints(point, np.zeros((3, 1)), np.zeros((3, 1)), intrinsic_matrix, dist_coeffs)[0]
    return tuple(point_2d[0, 0].astype(int))

def compute_w2c(c2w):
    rotation_inv = c2w[:3, :3].T
    translation_inv = -np.dot(rotation_inv, c2w[:3, 3])
    w2c = np.eye(4)
    w2c[:3, :3] = rotation_inv
    w2c[:3, 3] = translation_inv
    return w2c

def transform_world_to_camera():
    annotated = json.load(open("results.json"))
    centroids = np.array(annotated[0]["centroid"])
    vec = np.array(annotated[0]["vector"])
    transforms = json.load(open("./output_for_mt/transforms.json"))
    
    key = list(transforms.keys())[0]
    fx, fy = transforms[key]["camera_params"]["fx"][0], transforms[key]["camera_params"]["fy"][0]
    cx, cy = transforms[key]["camera_params"]["cx"][0], transforms[key]["camera_params"]["cy"][0]
    distortion_params = transforms[key]["camera_params"]["distortion_params"]

    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    distortion_params.extend([0, 0])
    dist_coeffs = np.array(distortion_params)
    # dist_coeffs = dist_coeffs[[True,True,False,False,True,True]]

    axis_length = 0.1  # Length of the axes
    x_axis = normalize(centroids[1] - centroids[0])
    y_axis = normalize(np.cross([0, 0, 1], x_axis))  # Assuming Z is up
    z_axis = np.cross(x_axis, y_axis)
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    axes = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
    rotated_axes = np.dot(rotation_matrix, axes.T).T

    results = []
    pbar = tqdm(transforms.items())
    pbar.set_description("Preprocess")
    for (key, transform) in pbar:
        c2w = np.eye(4)
        c2w[:3, :4] = np.array(transform["c2w"])
        # w2c = np.linalg.inv(c2w)
        w2c = compute_w2c(c2w)

        # Transform centroid and axes to camera coordinates
        centroid_cam = transform_vector(centroids[0], w2c)
        transformed_axes = [transform_vector(centroids[0] + axis, w2c) for axis in rotated_axes]

        image_number = int(key.split('_')[-1])
        image_path = f"output_for_mt/{image_number}.png"
        image = cv2.imread(image_path)

        # Project and draw axes
        centroid_image = project_to_image_plane(centroid_cam, intrinsic_matrix, dist_coeffs)
        for axis, color in zip(transformed_axes, [(0, 0, 255), (0, 0, 0), (0, 0, 0)]):
            axis_point_image = project_to_image_plane(axis, intrinsic_matrix, dist_coeffs)
            axis_point_image = np.clip(axis_point_image, 0, 2000)
            if DEBUG:
                image = cv2.line(image, centroid_image, axis_point_image, color, 2)

        if DEBUG:
            # Display the image
            cv2.imshow("3D Axes", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        results.append({
            "name": image_number,
            "image": image,
            "centroids": [transform_vector(c, w2c) for c in centroids]
        })
    return results
        

if __name__ == "__main__":
    transform_world_to_camera()
