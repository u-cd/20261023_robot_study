"""11点をZ=0、手首80度固定として実効運動学を推定。アームは動かさない。"""
import argparse
from datetime import datetime,timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/robot_matplotlib')
import numpy as np
from control.arm_kinematics import fit,forward,inverse_preview,PARAMETER_NAMES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,default=ROOT/'runs/arm_calibration/20261030_221230_241932')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/arm_kinematics')
    parser.add_argument('--starts',type=int,default=10)
    args=parser.parse_args()
    samples=[]
    for path in args.session.glob('*/sample.json'):
        sample=json.loads(path.read_text())
        samples.append((path,sample))
    samples.sort(key=lambda pair:pair[1]['point_number'])
    if len({r['point_number'] for _,r in samples})!=len(samples):
        raise ValueError('この初期評価は各点1記録を前提にしています。重複の扱いを指定してください。')
    angles=np.array([r['joint_command_degrees'] for _,r in samples],dtype=float)
    if not np.all(angles[:,3:]==[80,150]):raise ValueError('手首80度・グリッパー150度に限定したモデルです。')
    xy=np.array([r['reference_xyz_mm'][:2] for _,r in samples],dtype=float)
    positions=np.column_stack([xy,np.zeros(len(xy))])
    candidates=[]
    for signs in itertools.product([1,-1],repeat=2):
        candidate=fit(angles[:,:3],positions,signs,starts=args.starts)
        candidates.append(candidate)
        print('候補',signs,'RMSE',round(candidate['rmse_mm'],3),'mm',flush=True)
    best=min(candidates,key=lambda c:c['rmse_mm'])
    p=np.array(best['parameters']);signs=best['signs']
    predictions=forward(p,angles[:,:3],signs)
    held_out=[]
    for i in range(len(samples)):
        keep=np.arange(len(samples))!=i
        refit=fit(angles[keep,:3],positions[keep],signs,starts=3,initial=p,seed=42+i)
        prediction=forward(refit['parameters'],angles[i,:3],signs)[0]
        held_out.append({'point':samples[i][1]['point_number'],'predicted_xyz_mm':prediction.tolist(),
                         'error_mm':float(np.linalg.norm(prediction-positions[i])),
                         'bound_parameters':refit['bound_parameters']})
        print('除外予測',held_out[-1]['point'],round(held_out[-1]['error_mm'],3),'mm',flush=True)
    loo=np.array([r['error_mm'] for r in held_out])
    model={**best,'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),
        'model_type':'effective_base_yaw_plus_relative_two_link',
        'coordinate_frame':'calibration_rectangle','units':'mm','wrist_fixed_deg':80,
        'calibration_gripper_deg':150,'calculation_plane_z_mm':0,
        'height_assumption':'User authorized treating all 11 hover samples as Z=0. Raw height observations remain unknown.',
        'parameter_names':PARAMETER_NAMES,'reference_xy_mm':xy.tolist(),
        'reference_commands_deg':angles[:,:3].tolist(),
        'samples':[{'point':r['point_number'],'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'assumed_xyz_mm':position.tolist(),'predicted_xyz_mm':pred.tolist(),
                    'error_mm':float(np.linalg.norm(pred-position))}
                   for (path,r),position,pred in zip(samples,positions,predictions)],
        'candidate_fits':candidates,'leave_one_point_out':held_out,
        'leave_one_out_rmse_mm':float(np.sqrt(np.mean(loo**2))),
        'leave_one_out_max_error_mm':float(loo.max()),
        'leave_one_out_note':'Exploratory prediction at recorded commands. Same model family/signs selected using all samples; not an untouched test set or measured robot positioning error.',
        'motion_executed':False,'vertical_generalization_validated':False,
        'physical_parameters_identified':False,'ready_for_automatic_grasp':False,
        'limitations':['Narrow 0/30mm columns; one sample per point.',
                      'Z=0 is a task approximation, not measured clearance.',
                      'Servo command degrees assumed nominal; no encoder feedback.',
                      'Bounds and upper-link-up assumption are model priors, not measured dimensions.',
                      'A bound-hitting parameter means geometry/height should not be treated as identified.',
                      'Opening gripper changes reference point; not calibrated here.']}
    preview=inverse_preview(model,[15,75,0])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'model.json').write_text(json.dumps(model,ensure_ascii=False,indent=2)+'\n')
    (args.output/'preview_15_75_0.json').write_text(json.dumps(preview,ensure_ascii=False,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,6))
    axes[0].scatter(xy[:,0],xy[:,1],label='Reference points',color='#146c94')
    axes[0].quiver(xy[:,0],xy[:,1],predictions[:,0]-xy[:,0],predictions[:,1]-xy[:,1],angles='xy',scale_units='xy',scale=1,color='#dd7c21')
    for (_,r),pos in zip(samples,xy):axes[0].annotate(str(r['point_number']),pos,xytext=(4,4),textcoords='offset points')
    axes[0].set(xlabel='Paper X [mm]',ylabel='Paper Y [mm]',title='Fit XY residuals (arrows: predicted - reference)',xlim=(-15,45),ylim=(-15,165))
    axes[0].set_aspect('equal');axes[0].grid(alpha=.25)
    x=np.arange(len(samples));axes[1].bar(x-.2,best['errors_mm'],.4,label='Fit XYZ error')
    axes[1].bar(x+.2,loo,.4,label='Leave-one-point-out XYZ error')
    axes[1].set_xticks(x,[str(r['point_number']) for _,r in samples])
    axes[1].set(xlabel='Grid point',ylabel='Model residual [mm]',title='Model residuals, NOT measured movement errors')
    axes[1].legend();axes[1].grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(args.output/'fit_diagnostics.png',dpi=150);plt.close(fig)
    print(json.dumps({'fit_rmse_mm':best['rmse_mm'],'fit_max_mm':best['max_error_mm'],
        'leave_one_out_rmse_mm':model['leave_one_out_rmse_mm'],'leave_one_out_max_mm':model['leave_one_out_max_error_mm'],
        'signs':signs,'bound_parameters':best['bound_parameters'],'preview':preview},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
