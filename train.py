from ultralytics import YOLO


if __name__ == "__main__":
    # Load a pretrained YOLOv8s model
    model = YOLO("runs/detect/train/weights/last.pt")    

    # Train the model on the COCO dataset for 100 epochs
    train_results = model.train(
        data="coco.yaml",  # Path to dataset configuration file
        epochs=100,
        seed=69,
        resume=True,
        amp=False,
        dropout=0.1,
        plots=True
    )
