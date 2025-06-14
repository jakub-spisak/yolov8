from ultralytics.utils.benchmarks import benchmark

if __name__ == "__main__":
    # Benchmark on GPU
    benchmark(model="runs/detect/train/weights/best.pt", data="coco.yaml", imgsz=640, half=False, device=0)