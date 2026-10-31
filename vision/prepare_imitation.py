"""Extract proposed 15-input / 9-action BC data from saved pre-action frames. No robot I/O."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control.imitation_record import ACTIONS, PHASES

FEATURE_NAMES = ['sponge_u', 'sponge_v', 'target_u', 'target_v',
                 'sponge_confidence', 'target_confidence', 'sponge_visible', 'target_visible',
                 'base_command', 'shoulder_command', 'elbow_command', 'gripper_command'] + [f'phase_{i}' for i in range(len(PHASES))]


def features(detections, state, phase):
    if type(phase) is not int or not 0 <= phase < len(PHASES):
        raise ValueError('段階番号が不正です。')
    # Missing detections use zero coordinates/confidence and an explicit visibility flag.
    obj = [detections.get(name) for name in ('sponge', 'target')]
    vector = [v for item in obj for v in (item['center_normalized'] if item else [0., 0.])]
    vector += [item['confidence'] if item else 0. for item in obj]
    vector += [float(item is not None) for item in obj]
    joints = {j['joint']: j for j in state['joints']}
    vector += [joints[n]['angle']/180. for n in (1, 2, 3, 5)]
    vector += [float(i == phase) for i in range(len(PHASES))]
    return vector


def extract_episode(episode, weights, model):
    episode, weights = Path(episode), Path(weights)
    meta = json.loads((episode/'episode.json').read_text())
    if meta['outcome'] != 'success':
        raise ValueError('成功して保存したデモだけを使用してください。')
    if meta.get('phases') != PHASES:
        raise ValueError('段階の定義が現在と異なります。旧データの段階対応を確認してください。')
    names = set(model.names.values())
    if not {'sponge', 'target'} <= names:
        raise ValueError(f'検出クラスが一致しません: {names}')
    source = episode/'samples.jsonl'
    if not source.exists():
        raise ValueError('操作がまだ記録されていません。')
    rows = []
    import torch
    for line in source.read_text().splitlines():
        sample = json.loads(line)
        if sample['event'] != 'result' or not sample['accepted']:
            continue
        result = model.predict(str(episode/sample['image']), imgsz=320, conf=.25, device='cpu', verbose=False)[0]
        torch.set_num_threads(2)  # Ultralytics may reset this at first predictor setup.
        detections = {}
        for box in result.boxes:
            name = model.names[int(box.cls.item())]
            confidence = float(box.conf.item())
            if name not in ('sponge', 'target') or confidence <= detections.get(name, {}).get('confidence', -1):
                continue
            xywh = box.xywhn[0].tolist()
            detections[name] = {'center_normalized': xywh[:2], 'box_xyxy_px': box.xyxy[0].tolist(), 'confidence': confidence}
        rows.append({'sample_index': sample['index'], 'image': sample['image'], 'phase': sample['phase'],
                     'frame_sequence': sample['frame_sequence'], 'frame_captured_at': sample['frame_captured_at'],
                     'command_sent_at': sample['command_sent_at'],
                     'frame_to_command_s': (datetime.fromisoformat(sample['command_sent_at'])-datetime.fromisoformat(sample['frame_captured_at'])).total_seconds(),
                     'dt_since_previous_action_s': sample.get('dt_since_previous_action_s'),
                     'detections': detections, 'input': features(detections, sample['state_before'], sample['phase']),
                     'delta_deg': sample['delta_deg'],
                     'control_mode': sample.get('control_mode','step'),
                     'interval_seconds': sample.get('interval_seconds'),
                     'label': ACTIONS.index(sample['action'])})
    result = {'schema_version': 1, 'episode_id': episode.name, 'outcome': meta['outcome'],
              'default_training_eligible': meta['outcome'] == 'success',
              'weights': str(weights), 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
              'inference': {'imgsz': 320, 'confidence': .25, 'device': 'cpu'},
              'feature_names': FEATURE_NAMES, 'action_classes': ACTIONS, 'phases': PHASES, 'samples': rows}
    output = episode/'features.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return output, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--weights', type=Path, default=ROOT/'runs/sponge_comparison/yolo11n/weights/best.pt')
    args = parser.parse_args()
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(2)
    output, count = extract_episode(args.episode, args.weights, YOLO(args.weights))
    print(f'{count}サンプルを保存しました: {output}')
    print('分割はエピソード単位です。成功して保存したデモだけを使用します。')


if __name__ == '__main__':
    main()
