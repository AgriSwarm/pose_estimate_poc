import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans

class GUIEditor:
    def __init__(self) -> None:
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    def run(self,obj_path):
        # interactive picking
        mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)
        assert mesh.textures

        picked_indices = []
        while len(picked_indices) < 2:
            picked_points = self._pick_points(mesh)
            picked_indices = [pp.coord for pp in picked_points]

        # clastering
        vertices = np.asarray(picked_indices)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(vertices)
        centroids = kmeans.cluster_centers_

        # visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        cylinder = self._create_cylinder_between_points(centroids[0], centroids[1], color=[1, 0, 0])
        vis.add_geometry(cylinder)
        vis.add_geometry(mesh)
        vis.add_geometry(self.coordinate_frame)
        vis.run() 
        vis.destroy_window()

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
        vis.add_geometry(self.coordinate_frame)
        vis.run()
        vis.destroy_window()
        return vis.get_picked_points()
    
def main():
    editor = GUIEditor()
    editor.run("mesh.obj")

if __name__ == "__main__":
    main()