from pathlib import Path
import torch
from ultralytics import YOLO

image = Path("data/synthetic_v1/images/train/aw_00001.png")

print("CUDA:", torch.cuda.is_available())
print("Image exists:", image.exists())

if not image.exists():
    raise SystemExit("Image not found. Check the dataset path.")

model = YOLO("yolo26n.pt")

results = model.predict(
    source=str(image),
    imgsz=640,
    device=0,
    save=True,
    project="runs/detect",
    name="baseline",
    exist_ok=True,
    verbose=True,
)

for result in results:
    print("Detected boxes:", len(result.boxes))
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        print(model.names[class_id], round(confidence, 3))

print("Inference finished")
print("Output: runs/detect/baseline")
