"""Prepare all successful raw demonstrations; never starts model training."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

def source_hashes(folder, rows):
    files = {'episode.json', 'samples.jsonl', 'commands.jsonl'} | {r['image'] for r in rows}
    return {name: hashlib.file_digest((folder/name).open('rb'), 'sha256').hexdigest()
            for name in sorted(files) if (folder/name).exists()}


def resample(rows, hz):
    if not 0 < hz <= 50 or len(rows) < 2:
        raise ValueError('Need at least two samples and 0 < hz <= 50.')
    # Serial reads span an interval. Midpoint is an estimate, not hardware time.
    origin = rows[0]['state_start_ns']
    times = np.array([((r['state_start_ns']-origin)+(r['state_end_ns']-origin))/2e9 for r in rows])
    if not np.all(np.diff(times)>0):
        raise ValueError('Observation times must increase strictly.')
    if np.diff(times).max() > .25:
        raise ValueError('Observation gap exceeds 250ms; inspect raw data before interpolation.')
    angles = []
    for row in rows:
        joints = sorted(row['state']['joints'], key=lambda j:j['joint'])
        if not row['state']['connected'] or [j['joint'] for j in joints] != [1,2,3,4,5] or not all(j['active'] for j in joints):
            raise ValueError('Disconnected or inactive joints in source.')
        angles.append([j['angle'] for j in joints])
    angles=np.asarray(angles,dtype=np.float64)
    if not np.isfinite(angles).all():
        raise ValueError('Nonfinite source angles.')
    # Reserve one more grid observation for the final action target. No terminal padding.
    count=int(np.floor((times[-1]-times[0])*hz + 1e-9))
    if count < 1:
        raise ValueError('Recording too short for requested grid.')
    grid=times[0]+np.arange(count+1)/hz
    interp=np.column_stack([np.interp(grid,times,angles[:,j]) for j in range(5)])
    frame_map={}
    for r in rows:
        sequence=r['frame_sequence']
        value=(r['frame_captured_ns'],r['image'])
        if sequence in frame_map and frame_map[sequence] != value:
            raise ValueError('Frame sequence has inconsistent timestamp or path.')
        frame_map[sequence]=value
    frames=sorted([(seq,*v) for seq,v in frame_map.items()],key=lambda f:f[1])
    frame_times=np.array([(f[1]-origin)/1e9 for f in frames])
    chosen=np.searchsorted(frame_times,grid[:-1],side='right')-1
    if (chosen<0).any():
        raise ValueError('No preceding image for first grid observation.')
    frame_age=grid[:-1]-frame_times[chosen]
    if frame_age.max()>.25:
        raise ValueError('Image age exceeds 250ms on output grid.')
    qpos=np.deg2rad(interp[:-1]).astype('float32')
    action=np.deg2rad(interp[1:]).astype('float32')
    qvel=np.zeros_like(qpos)
    qvel[1:]=np.diff(qpos,axis=0)*hz
    return dict(qpos=qpos,action=action,qvel=qvel,time=grid[:-1]-times[0],
                frame_sequence=np.array([frames[i][0] for i in chosen]),
                images=[frames[i][2] for i in chosen],frame_age=frame_age,
                max_source_gap_s=float(np.diff(times).max()),origin_ns=origin,
                grid_origin_relative_s=float(times[0]))


def convert(source, output, hz):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a new output directory.')
    if output==source or source in output.parents:
        raise ValueError('Write derived files outside the raw recording folder.')
    meta=json.loads((source/'episode.json').read_text())
    if meta['status']!='success':
        raise ValueError('Only completed successful demonstrations are eligible.')
    rows=[json.loads(line) for line in (source/'samples.jsonl').read_text().splitlines()]
    hashes=source_hashes(source,rows)
    data=resample(rows,hz)
    output.mkdir(parents=True)
    temp=output/'episode_0.hdf5.partial'
    first=cv2.imread(str(source/data['images'][0]))
    if first is None:
        raise ValueError('First image is missing or invalid.')
    height,width=first.shape[:2]
    count=len(data['qpos'])
    with h5py.File(temp,'w') as f:
        f.attrs.update(sim=False, action_offset=0, dt=1/hz, fps=hz,
                       angle_source='commanded_not_measured', joint_units='radian_all_5_including_gripper',
                       joint_order=json.dumps(meta['config']['joint_order']),
                       action_semantics='next_grid_commanded_position_derived_from_demonstration',
                       observation_time='midpoint_of_PC_STATUS_request_interval',
                       source=str(source), conversion_version=1)
        for name in ('qpos','qvel'):
            f.create_dataset('observations/'+name,data=data[name])
        f.create_dataset('action',data=data['action'])
        f.create_dataset('timestamps/time_s',data=data['time'])
        f.create_dataset('timestamps/frame_sequence',data=data['frame_sequence'])
        f.create_dataset('timestamps/frame_age_s',data=data['frame_age'])
        images=f.create_dataset('observations/images/top',(count,height,width,3),dtype='uint8',
                                chunks=(1,height,width,3),compression='lzf')
        last_name,last_rgb=None,None
        for i,name in enumerate(data['images']):
            if name!=last_name:
                bgr=cv2.imread(str(source/name))
                if bgr is None or bgr.shape!=first.shape:
                    raise ValueError('Missing image or inconsistent dimensions: '+name)
                last_rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);last_name=name
            images[i]=last_rgb
    after=source_hashes(source,rows)
    if hashes!=after:
        raise RuntimeError('Source recording changed during conversion; do not use output.')
    temp.rename(output/'episode_0.hdf5')
    summary={'source':str(source),'source_sha256':hashes,'hz':hz,'dt':1/hz,
             'raw_samples':len(rows),'output_samples':count,'duration_s':count/hz,
             'max_source_gap_s':data['max_source_gap_s'],
             'image_age_ms_percentiles':np.percentile(data['frame_age']*1000,[50,95,100]).tolist(),
             'initial_angles_deg':np.rad2deg(data['qpos'][0]).tolist(),
             'source_unchanged':True,'single_demonstration_only':True,
             'limitations':['Commanded positions, not measured joint angles',
                            'Interpolated qpos and next-grid qpos targets, not measured physical motion',
                            'Image uses latest recorded frame at or before each grid time',
                            'Gripper uses radians pending confirmation of open/closed endpoints',
                            'Variable episode length; use the documented chunk-padding loader patch for batching',
                            '20Hz chunk100 represents 5 seconds, unlike official 50Hz chunk100=2 seconds'],
             'alignment':'action_offset=0 disables upstream legacy real-data -1 shift for this file only'}
    (output/'conversion.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_sha256'},ensure_ascii=False,indent=2))


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
