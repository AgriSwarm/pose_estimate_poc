import math
import glob
import json
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from tqdm import tqdm
import viser.transforms as vtf

class GUIEditor:
    def __init__(self,with_texture=True) -> None:
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.with_texture = with_texture

    def run(self,obj_path):
        # interactive picking
        mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=self.with_texture)
        assert mesh.textures

        transforms = json.load(open("./output_for_mt/transforms.json"))
        keys = sorted(list(transforms.keys()))
        print(keys[0])
        transforms = transforms[keys[0]]

        c2w = np.eye(4)
        c2w[:3,:4] = np.array(transforms["c2w"])
        # c2w[:3,3] /= 10
        print(c2w)
        self.coordinate_frame.transform(c2w)

        # picked_indices = []
        # while len(picked_indices) < 2:
        #     picked_points = self._pick_points(mesh)
        #     picked_indices = [pp.coord for pp in picked_points]

        # clastering
        # vertices = np.asarray(picked_indices)
        # kmeans = KMeans(n_clusters=2)
        # kmeans.fit(vertices)
        centroids = np.array([[0.01026475,-0.04470124,-0.24673722],
                            [0.0046543,-0.05070926,-0.3248854]])
        # visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1080, height=1920)
        cylinder = self._create_cylinder_between_points(centroids[0], centroids[1], color=[1, 0, 0])
        vis.add_geometry(cylinder)
        vis.add_geometry(mesh)
        vis.add_geometry(self.coordinate_frame)
        vis.run() 
        vis.destroy_window()
        return centroids

    def _create_cylinder_between_points(self, p1, p2, color, radius=0.001):
        distance = np.linalg.norm(p1 - p2)
        mid_point = (p1 + p2) / 2

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=distance)
        cylinder.paint_uniform_color(color)

        vec = p2 - p1
        vec /= np.linalg.norm(vec)
        z_axis = np.array([0.0, 0.0, 1.0])
        rotation_axis = np.cross(z_axis,vec)
        rotation_axis /= np.linalg.norm(rotation_axis) 
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, vec), -1.0, 1.0))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

        cylinder.rotate(rotation_matrix, center=cylinder.get_center())
        cylinder.translate(mid_point - cylinder.get_center())

        return cylinder

    def _pick_points(self,mesh):
        print("Showing mesh. Please click on the mesh to select points...")
        vis = o3d.visualization.VisualizerWithVertexSelection()
        vis.create_window()
        vis.add_geometry(mesh)
        # vis.add_geometry(self.coordinate_frame)
        vis.run()
        vis.destroy_window()
        return vis.get_picked_points()
    
class Annotator:
    def __init__(self) -> None:
        editor = GUIEditor(with_texture=True)
        self.editor = editor

    def run(self):
        result_path = Path(f"results.json")
        data, already_processed = [], []
        if result_path.exists():
            data = json.load(result_path.open("r"))
            already_processed = [d["name"] for d in data]

        obj_list = [Path(path) for path in glob.glob("flower2/*.obj") if str(path) not in already_processed]
        for obj_path in tqdm(obj_list):
            print("Process:",obj_path)
            centroids = self.editor.run(str(obj_path))
            euler = self._compute_euler_angles(centroids[0], centroids[1])
            print("centroid:",centroids)
            print("euler:",euler)
            data.append({
                "name": obj_path.name,
                "centroid": centroids.tolist(),
                "euler": euler
            })
            json.dump(data,result_path.open("w"))
            
    def _compute_euler_angles(self,vec1, vec2):
        x, y, z = [v2 - v1 for v1, v2 in zip(vec1, vec2)]
        pitch = math.atan2(-z, math.sqrt(x**2 + y**2))
        yaw = math.atan2(y, x)
        roll = 0
        return [math.degrees(x) for x in [yaw, pitch, roll]]

def main():
    annotator = Annotator()
    annotator.run()

if __name__ == "__main__":
    main()