"""Fine-tune one detector using the recorded experiment configuration."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.tools/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.tools/matplotlib'))
os.environ.setdefault('YOLO_OFFLINE', 'true')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['yolo11n', 'yolo11s', 'deim_n', 'deim_s', 'deimv2_s', 'deimv2_m'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/vision_experiment.json')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    import torch
    torch.set_num_threads(cfg['threads'])
    dataset = ROOT / cfg['dataset']
    output = ROOT / cfg['output'] / args.model
    if output.exists():
        raise SystemExit(f'Refusing to overwrite {output}; use a config with a new output.')
    output.mkdir(parents=True)
    (output / 'experiment.json').write_text(json.dumps(cfg, indent=2) + '\n')
    if args.model.startswith('yolo'):
        from ultralytics import YOLO, settings
        from ultralytics.utils import torch_utils
        # select_device('cpu') otherwise resets PyTorch to Ultralytics' default (8 here).
        torch_utils.NUM_THREADS = cfg['threads']
        settings.update({'sync': False})
        model = YOLO(str(ROOT / 'models' / f'{args.model}.pt'))
        model.train(data=str(dataset / 'dataset.yaml'), epochs=cfg['epochs'],
                    imgsz=cfg['image_size'], batch=cfg['batch_size'], device='cpu', workers=0,
                    project=str(output.parent), name=output.name, exist_ok=True,
                    seed=cfg['seed'], amp=False, plots=False, cache=False,
                    optimizer='AdamW', lr0=0.001, lrf=0.1, cos_lr=True, warmup_epochs=2,
                    mosaic=0, mixup=0, fliplr=0, translate=0.05, scale=0.1,
                    hsv_h=0.015, hsv_s=0.2, hsv_v=0.2, erasing=0, patience=cfg['epochs']+1)
    else:
        v2 = args.model.startswith('deimv2_')
        vendor = ROOT / ('reference/DEIMv2' if v2 else 'reference/DEIM')
        size = args.model[-1]
        # Start with the smoke-test configuration, replacing data, schedule and model size.
        deim = json.loads((dataset / 'deim.yml').read_text())
        deim['__include__'] = [str(vendor / (f'configs/deimv2/deimv2_dinov3_{size}_coco.yml' if v2 else f'configs/deim_dfine/deim_hgnetv2_{size}_coco.yml'))]
        deim.update(epoches=cfg['epochs'], output_dir=str(output),
                    checkpoint_freq=cfg['epochs'], warmup_iter=36,
                    flat_epoch=3, lr_gamma=0.1, no_aug_epoch=0)
        deim['optimizer'] = {'type': 'AdamW', 'lr': 0.0002, 'weight_decay': 0.0001,
            'params': [{'params': '^backbone\\.', 'lr': 0.00005},
                       {'params': '^(?!backbone\\.)(?=.*(?:norm|bn|bias)).*$', 'weight_decay': 0.0}]}
        for split in ('train', 'val'):
            loader = deim[f'{split}_dataloader']
            loader['total_batch_size'] = cfg['batch_size']
            loader['collate_fn']['base_size'] = cfg['image_size']
            for op in loader['dataset']['transforms']['ops']:
                if op['type'] == 'Resize':
                    op['size'] = [cfg['image_size']]*2
        deim['eval_spatial_size'] = [cfg['image_size']]*2
        deim['train_dataloader']['dataset']['transforms']['ops'].insert(0, {'type': 'RandomPhotometricDistort', 'p': 0.5})
        if v2:
            # Full COCO detector weights supply the distilled backbone; no teacher download.
            deim['DINOv3STAs'] = {'weights_path': None}
            deim['DEIMCriterion'] = {'matcher': {'matcher_change_epoch': 15}}
            for split in ('train', 'val'):
                loader = deim[f'{split}_dataloader']
                loader['collate_fn']['copyblend_prob'] = 0.0
                ops = loader['dataset']['transforms']['ops']
                index = next(i for i, op in enumerate(ops) if op['type'] == 'ConvertPILImage') + 1
                ops.insert(index, {'type': 'Normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
        config = output / 'deim.yml'
        # PyYAML 1.1 treats JSON's "5e-05" notation as a string; emit proper YAML floats.
        import yaml
        config.write_text(yaml.safe_dump(deim, sort_keys=False))
        sys.path.insert(0, str(vendor))
        sys.argv = [str(vendor / 'train.py'), '-c', str(config), '-d', 'cpu', '--seed', str(cfg['seed']),
                    '-t', str(ROOT / (f'models/deimv2_dinov3_{size}_coco.pth' if v2 else f'models/deim_dfine_{size}_coco.pth'))]
        runpy.run_path(str(vendor / 'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
