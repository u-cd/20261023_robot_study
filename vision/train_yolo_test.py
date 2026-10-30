"""CPU-only one-epoch fine-tuning test. Does not connect to the robot."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.tools/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.tools/matplotlib'))
os.environ.setdefault('YOLO_OFFLINE', 'true')
import torch
from ultralytics import YOLO, settings

if __name__ == '__main__':
    torch.set_num_threads(4)
    settings.update({'sync': False})
    model = YOLO(str(ROOT / 'models/yolo11n.pt'))
    model.train(data=str(ROOT / 'data/sponge_training_test/dataset.yaml'),
                epochs=1, imgsz=320, batch=2, device='cpu', workers=0,
                project=str(ROOT / 'runs/training_test'), name='yolo11n',
                seed=42, amp=False, plots=False, cache=False,
                mosaic=0, fliplr=0, translate=0, scale=0, erasing=0)
