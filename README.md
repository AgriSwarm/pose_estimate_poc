# Flower Pose Estimation Model

Given a flower image cropped from an object detection bounding box, this model predicts its pose in Euler angles.

![Screenshot 2025-01-16 at 3 50 27](https://github.com/user-attachments/assets/708dd44c-d32e-4f18-b342-80d50316c640)


---

**Setup:**
```bash
cd mlcore/sixdrepnet
pip install poetry
poetry install
```

**Train:**
```bash
poetry run python main.py
```

**Predict:**
```bash
poetry run python inference.py
```

Or in Python:
```python
from inference import predict
results = predict("path/to/images")
print(results)
```
