"""Reproduce YOLOv8n training for the initial-detection ACT input."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/robot-yolo-train')
os.environ.setdefault('MPLCONFIGDIR','/tmp/robot-detector-mpl')
os.environ['MPLBACKEND']='Agg'
os.environ['OMP_NUM_THREADS']='2'
import torch

sys.stdout.reconfigure(line_buffering=True)
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['yolo'],default='yolo')
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output=args.output.resolve()
    if args.output.exists():raise SystemExit('Choose a new run directory.')
    root=ROOT/'data/initial_object_labels'
    manifest=json.loads((root/'manifest.json').read_text())
    for entry in manifest['images']:
        record=json.loads((root/'annotations'/f"{entry['id']}.json").read_text())
        if not record.get('reviewed'):raise ValueError('Unreviewed image: '+entry['id'])
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    started=time.perf_counter()
    from ultralytics import YOLO
    import ultralytics.utils.torch_utils as tu
    tu.NUM_THREADS=2
    model=YOLO(str(ROOT/'models/yolov8n_coco.pt'))
    model.train(data=str(root/'dataset.yaml'),epochs=args.epochs,imgsz=640,batch=4,
                device='cpu',workers=0,seed=0,patience=15,project=str(args.output.parent),
                name=args.output.name,exist_ok=False,plots=False,amp=False,cache=False,
                mosaic=0.5,close_mosaic=10,save_period=-1)
    summary=dict(model=args.model,elapsed_s=time.perf_counter()-started,
                 max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 torch=torch.__version__,threads=torch.get_num_threads(),epochs_requested=args.epochs,
                 data=str(root),train_images=40,validation_images=10,seed=0,
                 initial_weights='models/yolov8n_coco.pt')
    (args.output/'runtime.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
