import torch
from pytorch_lightning import Trainer
import random
import numpy as np
from module import MitoDataModule, SixDRepNetModule
from PIL import Image

def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])

def predict(image_path):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)
    
    checkpoint_path = 'checkpoints/best.ckpt'

    data_module = MitoDataModule(data_path='mito_datasets/dataset.json')
    data_module.setup(stage='test')

    model = SixDRepNetModule.load_from_checkpoint(checkpoint_path)
    model = model.cuda()
    model.eval() 

    image = Image.open(image_path)
    image = data_module.transform(image).unsqueeze(0).cuda()
    
    with torch.no_grad():
        R = model(image)

    R_np = R.cpu().numpy()[0]
    euler = rotation_matrix_to_euler_angles(R_np)
    
    return euler

results = predict("mito_datasets/flower_normal_x0_y0_z0.png")
print("Euler Angles (Degrees):", results)