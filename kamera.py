from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    results = model(source="video.mp4", show=True, conf=0.4, save=True)