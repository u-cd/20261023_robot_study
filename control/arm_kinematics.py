"""台座＋平面2リンクの実効モデル。計算のみで、シリアル通信は行わない。"""
import itertools
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import Delaunay

PARAMETER_NAMES=['base_x_mm','base_y_mm','base_yaw_rad','shoulder_height_mm',
                 'effective_upper_length_mm','effective_distal_length_mm',
                 'shoulder_offset_rad','elbow_offset_rad']


def forward(parameters, commands_deg, signs=(-1,1)):
    q=np.deg2rad(np.atleast_2d(np.asarray(commands_deg,dtype=float)))
    if q.shape[1]!=3 or not np.isfinite(q).all():
        raise ValueError('台座・肩・肘の3指令を指定してください。')
    bx,by,yaw,height,l1,l2,a0,b0=np.asarray(parameters,dtype=float)
    a=signs[0]*q[:,1]+a0
    b=a+signs[1]*q[:,2]+b0
    radius=l1*np.cos(a)+l2*np.cos(b)
    theta=q[:,0]+yaw
    return np.column_stack([bx+radius*np.cos(theta),by+radius*np.sin(theta),
                            height+l1*np.sin(a)+l2*np.sin(b)])


def bounds(commands, signs):
    shoulder=signs[0]*np.deg2rad(np.asarray(commands)[:,1])
    # Upper link points above its shoulder across the observed poses.
    low=[-400,-150,-2.5,15,40,40,-shoulder.min(),-2*np.pi]
    high=[0,350,0,220,250,300,np.pi-shoulder.max(),2*np.pi]
    return np.array(low),np.array(high)


def fit(commands, positions, signs=(-1,1), starts=10, seed=42, initial=None):
    commands=np.asarray(commands,dtype=float);positions=np.asarray(positions,dtype=float)
    if commands.ndim!=2 or commands.shape[1]!=3 or positions.shape!=commands.shape or len(commands)<5:
        raise ValueError('5点以上の指令とXYZ位置が必要です。')
    if not np.isfinite(commands).all() or not np.isfinite(positions).all():
        raise ValueError('指令と位置は有限値にしてください。')
    low,high=bounds(commands,signs)
    rng=np.random.default_rng(seed)
    guesses=[]
    if initial is not None:guesses.append(np.clip(initial,low+1e-8,high-1e-8))
    while len(guesses)<starts:
        a0=rng.uniform(low[6],high[6])
        a=signs[0]*np.deg2rad(commands[:,1].mean())+a0
        b0=(-rng.uniform(.2,1.5)-a-signs[1]*np.deg2rad(commands[:,2].mean())+np.pi)%(2*np.pi)-np.pi
        guesses.append([-105,100,-1.35,70,100,160,a0,b0])
    best=None
    for guess in guesses:
        result=least_squares(lambda p:(forward(p,commands,signs)-positions).ravel(),guess,
            bounds=(low,high),x_scale='jac',max_nfev=2000,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        if best is None or np.sum(result.fun**2)<np.sum(best.fun**2):best=result
    errors=np.linalg.norm(best.fun.reshape(-1,3),axis=1)
    scales=np.linalg.norm(best.jac,axis=0)
    singular=np.linalg.svd(best.jac/np.maximum(scales,1e-12),compute_uv=False)
    near_bound=[name for i,name in enumerate(PARAMETER_NAMES)
                if min(abs(best.x[i]-low[i]),abs(best.x[i]-high[i]))<1e-4]
    return {'parameters':best.x.tolist(),'signs':list(signs),'errors_mm':errors.tolist(),
            'rmse_mm':float(np.sqrt(np.mean(errors**2))),'max_error_mm':float(errors.max()),
            'jacobian_rank':int(np.linalg.matrix_rank(best.jac)),
            'column_normalized_condition_number':float(singular[0]/singular[-1]),
            'bound_parameters':near_bound,'optimizer_success':bool(best.success)}


def inverse_preview(model,xyz,allow_unvalidated_height=False):
    target=np.asarray(xyz,dtype=float)
    if target.shape!=(3,) or not np.isfinite(target).all():
        raise ValueError('有限なXYZを指定してください。')
    samples=np.array(model['reference_xy_mm'])
    if Delaunay(samples).find_simplex(target[:2])<0:
        raise ValueError('11点が囲む領域の外側です。外挿は行いません。')
    if abs(target[2])>1e-9 and not allow_unvalidated_height:
        raise ValueError('高さ方向は未検証です。通常の候補計算はZ=0だけに限定します。')
    known=np.asarray(model['reference_commands_deg'])
    order=np.argsort(np.linalg.norm(samples-target[:2],axis=1))
    best=None
    for index in order[:4]:
        result=least_squares(lambda q:(forward(model['parameters'],q,model['signs'])[0]-target),
            known[index],bounds=([0,0,0],[180,180,180]),max_nfev=500,xtol=1e-11,ftol=1e-11,gtol=1e-11)
        if best is None or np.linalg.norm(result.fun)<np.linalg.norm(best.fun):best=result
    if np.linalg.norm(best.fun)>.1:
        raise ValueError('モデル上で指定位置に到達する解がありません。')
    rounded=np.rint(best.x).astype(int)
    predicted=forward(model['parameters'],rounded,model['signs'])[0]
    return {'target_xyz_mm':target.tolist(),'command_degrees_float':best.x.tolist(),
            'command_degrees_rounded':rounded.tolist(),
            'model_predicted_xyz_after_rounding_mm':predicted.tolist(),
            'model_rounding_error_mm':float(np.linalg.norm(predicted-target)),
            'motion_executed':False,'physical_reach_and_path_verified':False,
            'height_generalization_verified':False}
