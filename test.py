from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("runs/detect/train/weights/last.pt")  # load a custom model
    
    # Validate the model
    metrics = model.val(data="coco.yaml", split="test", plots=True, project="runs/detect/train/test", name="2", verbose=True)
