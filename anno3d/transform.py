import math
import glob
import json
import cv2
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from tqdm import tqdm

def transform_vector(vec, transform_matrix):
    vec_homogeneous = np.append(vec, 1)
    transformed_vec_homogeneous = np.dot(transform_matrix, vec_homogeneous)
    return transformed_vec_homogeneous[:3] / transformed_vec_homogeneous[3]

def project_to_image_plane(point, intrinsic_matrix, dist_coeffs):
    point = point.reshape(1, 1, -1)
    point_2d = cv2.projectPoints(point, np.zeros((3, 1)), np.zeros((3, 1)), intrinsic_matrix, dist_coeffs)[0]
    return tuple(point_2d[0, 0].astype(int))

def main():
    annotated = json.load(open("results.json"))
    centroids = np.array(annotated[0]["centroid"])
    transforms = json.load(open("./output_for_mt/transforms.json"))
    
    # Load camera intrinsic matrix (replace with actual values)
    scale_w, scale_h = 540 / 1080, 960 / 1920
    fl_x, fl_y = 1370.6741302239604, 1367.882226657102
    cx, cy = 540, 960
    fl_x, fl_y = fl_x * scale_w, fl_y * scale_h
    cx, cy = cx * scale_w, cy * scale_h
    intrinsic_matrix = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
    dist_coeffs = np.array([0.05255798683457146, -0.06900366039374681, 0.000966278511558134, 0.0013915894078101285])

    for (key, transform) in transforms.items():
        c2w = np.eye(4)
        c2w[:3,:4] = np.array(transform["c2w"])
        w2c = np.linalg.inv(c2w) 

        # Transform centroids to camera coordinates
        centroid1_cam = transform_vector(centroids[1], w2c)
        centroid2_cam = transform_vector(centroids[0], w2c)

        # Project points onto image plane
        start_point = project_to_image_plane(centroid1_cam, intrinsic_matrix, dist_coeffs)
        end_point = project_to_image_plane(centroid2_cam, intrinsic_matrix, dist_coeffs)


        image_number = int(key.split('_')[-1])
        image_path = f"output_for_mt/{image_number}.png"
        image = cv2.imread(image_path)
        print(start_point, end_point)
        image = cv2.arrowedLine(image, start_point, end_point, (0, 0, 255), 5, tipLength=0.3)

        # Display the image
        cv2.imshow("Flower with Vector", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
