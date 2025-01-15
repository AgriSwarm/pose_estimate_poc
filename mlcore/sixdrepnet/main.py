import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from module import MitoDataModule, SixDRepNetModule
import random
import numpy as np

torch.random.initial_seed()  
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)

# Initialize DataModule
data_module = MitoDataModule(data_path='mito_datasets/dataset.json')

# Initialize model
model = SixDRepNetModule()

# Setup checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)

# Initialize Trainer
trainer = pl.Trainer(max_epochs=30, callbacks=[checkpoint_callback])

# Train the model
trainer.fit(model, datamodule=data_module)

# Evaluate on test set
trainer.validate(model, datamodule=data_module)

print("Training complete.")