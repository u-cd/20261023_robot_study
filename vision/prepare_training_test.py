"""Validate a snapshot and create matching YOLO/COCO splits; originals stay intact."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import shutil

import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/sponge')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/sponge_training_test')
    parser.add_argument('--val-session', help='Hold out an entire capture session for validation')
    parser.add_argument('--test-sessions', nargs='*', default=[], help='Entire sessions reserved for final evaluation')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f'Output already exists: {output}. Use a new --output directory.')
    records = {Path(r['image']).name: r for r in map(json.loads, (source / 'samples.jsonl').read_text().splitlines())}
    images = sorted((source / 'images/train').glob('*.jpg'))
    sessions = collections.Counter(records[p.name]['session'] for p in images)
    if len(sessions) < 2:
        raise SystemExit('Need at least two capture sessions for a session-separated split.')
    val_session = args.val_session or min(sessions, key=lambda s: (abs(sessions[s] - len(images) * .2), s))
    if val_session not in sessions or not set(args.test_sessions) <= sessions.keys() or val_session in args.test_sessions:
        raise ValueError('Invalid or overlapping validation/test sessions')
    if set(sessions) <= {val_session, *args.test_sessions}:
        raise ValueError('No training sessions remain')
    verified, hashes = [], set()
    for image in images:
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError(f'Unreadable image: {image}')
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f'Duplicate image bytes: {image}')
        hashes.add(digest)
        h, w = frame.shape[:2]
        label = source / 'labels/train' / (image.stem + '.txt')
        boxes = []
        for line in label.read_text().splitlines():
            values = list(map(float, line.split()))
            if len(values) != 5:
                raise ValueError(f'Invalid label: {label}')
            cls, x, y, bw, bh = values
            if cls not in (0, 1) or not all(math.isfinite(v) for v in values):
                raise ValueError(f'Invalid class/numbers: {label}')
            if not (bw > 0 and bh > 0 and x-bw/2 >= -1e-5 and y-bh/2 >= -1e-5 and x+bw/2 <= 1.00001 and y+bh/2 <= 1.00001):
                raise ValueError(f'Box outside image: {label}')
            boxes.append((int(cls), [(x-bw/2)*w, (y-bh/2)*h, bw*w, bh*h]))
        session = records[image.name]['session']
        split = 'val' if session == val_session else 'test' if session in args.test_sessions else 'train'
        verified.append((image, label, w, h, boxes, split, digest))
    output.mkdir(parents=True)
    manifest = {'purpose': 'Session-separated local experiment; limited to the current camera and scene', 'validation_session': val_session, 'test_sessions': args.test_sessions, 'sessions': dict(sessions), 'images': []}
    categories = [{'id': 0, 'name': 'sponge'}, {'id': 1, 'name': 'target'}]
    splits = ('train', 'val', 'test') if args.test_sessions else ('train', 'val')
    for split in splits:
        coco = {'info': {'description': 'Sponge trainability test'}, 'licenses': [], 'images': [], 'annotations': [], 'categories': categories}
        for image, label, w, h, boxes, part, digest in verified:
            if part != split:
                continue
            for kind, src in [('images', image), ('labels', label)]:
                dest = output / kind / split / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            image_id = len(coco['images']) + 1
            coco['images'].append({'id': image_id, 'file_name': image.name, 'width': w, 'height': h})
            for cls, box in boxes:
                coco['annotations'].append({'id': len(coco['annotations']) + 1, 'image_id': image_id, 'category_id': cls, 'bbox': box, 'area': box[2]*box[3], 'iscrowd': 0})
            manifest['images'].append({'name': image.name, 'split': split, 'sha256': digest, 'session': records[image.name]['session']})
        ann = output / 'annotations' / f'{split}.json'
        ann.parent.mkdir(exist_ok=True)
        ann.write_text(json.dumps(coco, indent=2) + '\n')
        print(f'{split}: {len(coco["images"])} images, {len(coco["annotations"])} boxes')
    dataset = {'path': str(output), 'train': 'images/train', 'val': 'images/val', 'names': {0: 'sponge', 1: 'target'}}
    if args.test_sessions:
        dataset['test'] = 'images/test'
    (output / 'dataset.yaml').write_text(json.dumps(dataset, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    config = {
        '__include__': [str(ROOT / 'reference/DEIM/configs/deim_dfine/deim_hgnetv2_n_coco.yml')],
        'num_classes': 2, 'remap_mscoco_category': False,
        'device': 'cpu', 'epoches': 1, 'eval_spatial_size': [320, 320],
        'output_dir': str(ROOT / 'runs/training_test/deim_n'),
        'use_amp': False, 'use_ema': False, 'sync_bn': False,
        'HGNetv2': {'pretrained': False}, 'warmup_iter': 1,
        'flat_epoch': 1, 'no_aug_epoch': 0, 'print_freq': 5, 'checkpoint_freq': 1,
    }
    for split in ('train', 'val'):
        ops = [{'type': 'Resize', 'size': [320, 320]}, {'type': 'ConvertPILImage', 'dtype': 'float32', 'scale': True}]
        if split == 'train':
            ops.append({'type': 'ConvertBoxes', 'fmt': 'cxcywh', 'normalize': True})
        config[f'{split}_dataloader'] = {
            'total_batch_size': 2, 'num_workers': 0, 'drop_last': False,
            'dataset': {'img_folder': str(output / 'images' / split), 'ann_file': str(output / 'annotations' / f'{split}.json'),
                        'transforms': {'ops': ops, 'policy': None, 'mosaic_prob': 0}},
            'collate_fn': {'base_size': 320, 'base_size_repeat': None, 'stop_epoch': 1000, 'mixup_prob': 0},
        }
    (output / 'deim.yml').write_text(json.dumps(config, indent=2) + '\n')
    print(f'Prepared: {output}')


if __name__ == '__main__':
    main()
