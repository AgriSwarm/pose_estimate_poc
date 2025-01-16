**Flower Pose Estimation Model**

Given a flower image cropped from an object detection bounding box, this model predicts its pose in Euler angles.

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
