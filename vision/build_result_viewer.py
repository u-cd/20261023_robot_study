"""Build an offline, self-contained viewer from saved images and detector predictions."""
import base64
import json
from pathlib import Path

from evaluate_detectors import iou

ROOT = Path(__file__).resolve().parents[1]


def build_data():
    cfg = json.loads((ROOT / 'configs/vision_experiment.json').read_text())
    output, dataset = ROOT / cfg['output'], ROOT / cfg['dataset']
    data = {'models': cfg['models'], 'confidence': cfg['confidence_threshold'], 'splits': {}}
    for split in ('test', 'val'):
        annotations = json.loads((dataset / f'annotations/{split}.json').read_text())
        report = json.loads((output / f'{split}_evaluation.json').read_text())
        predictions = {name: json.loads((output / name / f'{split}_predictions.json').read_text()) for name in cfg['models']}
        frames = []
        for image in annotations['images']:
            image_id = image['id']
            truth = []
            for row in annotations['annotations']:
                if row['image_id'] == image_id:
                    x, y, w, h = row['bbox']
                    truth.append({'cls': row['category_id'], 'box': [x, y, x+w, y+h]})
            frame = {'id': image_id, 'name': image['file_name'], 'truth': truth,
                'image': 'data:image/jpeg;base64,' + base64.b64encode((dataset / f'images/{split}' / image['file_name']).read_bytes()).decode(), 'models': {}}
            for name in cfg['models']:
                rows = sorted([r for r in predictions[name] if r['image_id'] == image_id and r['score'] >= cfg['confidence_threshold']], key=lambda r: r['score'], reverse=True)
                matched, boxes = set(), []
                for row in rows:
                    x, y, w, h = row['bbox']
                    box = [x, y, x+w, y+h]
                    candidates = [(iou(box, g['box']), j) for j, g in enumerate(truth) if j not in matched and g['cls'] == row['category_id']]
                    overlap, index = max(candidates, default=(0, -1))
                    ok = overlap >= cfg['iou_threshold']
                    if ok:
                        matched.add(index)
                    boxes.append({'cls': row['category_id'], 'box': box, 'score': row['score'], 'ok': ok})
                measured = next(r for r in report['models'][name]['per_image'] if r['image_id'] == image_id)
                assert len(matched) == measured['matched']
                assert sum(not b['ok'] for b in boxes) == measured['false_positives']
                frame['models'][name] = {'boxes': boxes, 'missed': [g for j, g in enumerate(truth) if j not in matched], 'tp': len(matched), 'fp': measured['false_positives'], 'fn': len(truth)-len(matched)}
            frames.append(frame)
        data['splits'][split] = {'frames': frames, 'summary': {name: {key: report['models'][name][key] for key in ('map50_95', 'latency_mean_ms')} for name in cfg['models']}}
    return output, data


def main():
    output, data = build_data()
    template = (ROOT / 'vision/result_viewer.html').read_text()
    document = template.replace('/*__RESULT_DATA__*/null', json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
    path = output / 'viewer.html'
    path.write_text(document)
    print(f'Created {path}: 12 test + 12 validation images, {len(data["models"])} models. No server required.')


if __name__ == '__main__':
    main()
