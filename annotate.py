import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
from math import cos, sin
from copy import deepcopy
from pathlib import Path

class GUI:
    def __init__(self) -> None:
        self.image_label = None
        self.yaw, self.pitch, self.roll = 0, 0, 0
        self.mouse_pressed = False
        self.prev_x, self.prev_y = 0, 0

    def build(self):
        root = tk.Tk()
        root.title("Head Pose Annotation Tool")

        # Sliders for yaw, pitch, and roll
        yaw_slider = tk.Scale(
            root,
            from_=-180,
            to=180,
            orient='horizontal',
            label='Yaw')
        yaw_slider.pack()
        pitch_slider = tk.Scale(
            root,
            from_=-180,
            to=180,
            orient='horizontal',
            label='Pitch')
        pitch_slider.pack()
        roll_slider = tk.Scale(
            root,
            from_=-180,
            to=180,
            orient='horizontal',
            label='Roll')
        roll_slider.pack()

        # Button to update the image
        update_button = tk.Button(
            root,
            text="Update Image",
            command=self.update_image)
        update_button.pack()

        next_button = tk.Button(
            root,
            text="Next",
            command=self.save_euler_angle)
        next_button.pack()

        # Load your initial image here
        self.root = root
        self.yaw_slider = yaw_slider
        self.pitch_slider = pitch_slider
        self.roll_slider = roll_slider

        #find image file using glob
        image_widget = self.set_next_image()

        # Add event bindings to the canvas or image display widget
        image_widget.bind("<Button-1>", self.mouse_down)  # Mouse button down
        image_widget.bind(
            "<ButtonRelease-1>",
            self.mouse_up)  # Mouse button up
        # Mouse move with button pressed
        image_widget.bind("<B1-Motion>", self.mouse_move)

        root.mainloop()

    def mouse_down(self, event):
        self.mouse_pressed = True
        self.prev_x, self.prev_y = event.x, event.y

    def mouse_up(self, event):
        self.mouse_pressed = False

    def mouse_move(self, event):
        delta = 0.5
        if self.mouse_pressed:
            dx, dy = event.x - self.prev_x, event.y - self.prev_y
            self.yaw += dx * delta
            self.pitch += dy * delta
            self.update_image()
            self.prev_x, self.prev_y = event.x, event.y
            self.yaw_slider.set(self.yaw)
            self.pitch_slider.set(self.pitch)
            self.roll_slider.set(self.roll)

    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx is not None and tdy is not None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll)
                     * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch)
                     * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 4)

        return img

    def set_image(self, path):
        self.image_path = path
        self.image = cv2.imread(str(path))
        return self.image is not None
        
    def get_annotated_set(self):
        annotations_dir = Path("anno")
        annotations_dir.mkdir(exist_ok=True)
        anno_file = annotations_dir / Path("annotations.json")

        data = []
        if anno_file.exists():
            data = json.load(anno_file.open())
            
        return set([d["image_path"] for d in data])

    def save_euler_angle(self):
        annotations_dir = Path("anno/")
        annotations_dir.mkdir(exist_ok=True)
        
        anno_file = annotations_dir / Path("annotations.json")
        if anno_file.exists():
            data = json.load(anno_file.open())
        else:
            data = []
        data.append({
            "image_path": str(self.image_path),
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll
        })
        json.dump(data, anno_file.open("w"))
        self.set_next_image()
    
    def set_next_image(self):
        images = list(Path("images").glob("*"))
        annotated_list = self.get_annotated_set()
        image_paths = [img for img in images if str(img) not in annotated_list]
        for i in range(len(image_paths)):
            image_path = image_paths[i]
            suc = self.set_image(image_path)
            if suc:
                break
        assert suc, "No image found"
        image_widget = self.update_image()
        return image_widget

    def update_image(self):
        # Get yaw, pitch, roll values from sliders
        yaw = self.yaw_slider.get()
        pitch = self.pitch_slider.get()
        roll = self.roll_slider.get()
        # Update the image with new axes
        annotated_img = self.draw_axis(deepcopy(self.image), yaw, pitch, roll)
        # Display updated image
        self.display_image(annotated_img)
        return self.image_label

    def display_image(self, cv_img):
        # Convert the color scheme from BGR (OpenCV) to RGB (Tkinter)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # Convert the OpenCV image to PIL format
        pil_img = Image.fromarray(cv_img)
        # Convert the PIL image to ImageTk format
        tk_img = ImageTk.PhotoImage(pil_img)

        # Check if the image label already exists
        if self.image_label is not None:
            # If it exists, update the image
            self.image_label.config(image=tk_img)
            self.image_label.image = tk_img
        else:
            # If it does not exist, create a new label and add it to the GUI
            self.image_label = tk.Label(self.root, image=tk_img)
            self.image_label.image = tk_img
            self.image_label.pack()


def main():
    gui = GUI()
    gui.build()


if __name__ == "__main__":
    main()
