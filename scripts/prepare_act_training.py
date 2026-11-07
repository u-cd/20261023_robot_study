"""Prepare all successful raw demonstrations; never starts model training."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np

if __package__:
    from .prepare_act_dataset import convert
else:
    from prepare_act_dataset import convert


def prepare(source, output, expected=50, hz=20):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError('Choose a new derived-data directory outside the raw recordings.')
    folders = sorted(p.parent for p in source.glob('*/episode.json')
                     if json.loads(p.read_text())['status'] == 'success')
    if len(folders) != expected:
        raise ValueError(f'Expected {expected} successful episodes; found {len(folders)}')
    output.mkdir(parents=True)
    (output / 'provenance').mkdir()
    order = np.random.RandomState(1).permutation(len(folders))
    train_ids = set(order[:int(.8 * len(folders))].tolist())
    records = []
    for i, folder in enumerate(folders):
        work = output / f'.converting_{i}'
        with contextlib.redirect_stdout(io.StringIO()):
            convert(folder, work, hz)
        dest = output / f'episode_{i}.hdf5'
        (work / 'episode_0.hdf5').rename(dest)
        provenance = output / 'provenance' / f'episode_{i}.json'
        (work / 'conversion.json').rename(provenance)
        work.rmdir()
        with h5py.File(dest, 'r') as f:
            count = len(f['action'])
            record = {'episode_id': i, 'source': folder.name, 'file': dest.name,
                      'split': 'train' if i in train_ids else 'validation',
                      'steps': count, 'duration_s': count / hz,
                      'image_shape': list(f['observations/images/top'].shape),
                      'qpos_shape': list(f['observations/qpos'].shape),
                      'action_shape': list(f['action'].shape),
                      'initial_angles_deg': np.rad2deg(f['observations/qpos'][0]).tolist(),
                      'bytes': dest.stat().st_size,
                      'sha256': hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest(),
                      'provenance': str(provenance.relative_to(output))}
        records.append(record)
        print(f'{i + 1}/{len(folders)}: {folder.name} -> {dest.name}, {count} steps', flush=True)
    manifest = {'status': 'prepared_not_trained', 'hz': hz, 'source': str(source),
                'episodes': records, 'total_steps': sum(r['steps'] for r in records),
                'total_bytes': sum(r['bytes'] for r in records),
                'split_seed': 1, 'training_seed': 0,
                'train_ids_in_official_order': order[:int(.8 * len(folders))].tolist(),
                'validation_ids_in_official_order': order[int(.8 * len(folders)):].tolist(),
                'raw_recordings_modified': False,
                'processing': '20Hz alignment; no pause removal or time warping',
                'angle_source': 'commanded_not_measured', 'action_offset': 0}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/act_demonstrations_raw'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected', type=int, default=50)
    parser.add_argument('--hz', type=float, default=20)
    args = parser.parse_args()
    prepare(args.source, args.output, args.expected, args.hz)
