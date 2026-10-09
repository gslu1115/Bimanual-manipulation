"""Offline optical-depth residuals against fixed planar station surfaces.

Uses saved RGB-D/calibration and STATIC geometry only. It is a calibration
diagnostic, not a robot-path acceptance test or a carton truth input.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from workstation.simulation.station_priors import depth_backboards


def board_depth(K,T,shape,boards):
    v,u=np.indices(shape);cv=np.stack(((u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones(shape)),axis=-1)
    direction=cv@T[:3,:3].T;origin=T[:3,3]
    depth=np.full(shape,np.inf)
    for board in boards:
        lower=np.asarray(board.centre)-np.asarray(board.size)/2
        upper=np.asarray(board.centre)+np.asarray(board.size)/2
        safe=np.where(np.abs(direction)>1e-12,direction,np.where(direction<0,-1e-12,1e-12))
        a=(lower-origin)/safe;b=(upper-origin)/safe
        near=np.maximum(np.minimum(a,b).max(axis=-1),0.)
        far=np.maximum(a,b).min(axis=-1)
        candidate=np.where((far>=near)&(near>0.),near,np.inf)
        depth=np.minimum(depth,candidate)
    return depth


def audit_frame(data,boards):
    import cv2
    predicted=board_depth(data['K'],data['T_workcell_from_camera_cv'],data['depth_m'].shape,boards)
    valid=data['valid_depth']&np.isfinite(predicted)
    residual=np.zeros(predicted.shape);residual[valid]=data['depth_m'][valid]-predicted[valid]
    # A board prediction can be occluded by robots/table/cartons. Foreground
    # returns are explicitly excluded, not relabelled empty. Retain up to
    # 50 mm of residual so the diagnostic cannot silently trim a 3 mm error.
    foreground=valid&(residual<-.05)
    inconsistent=valid&(np.abs(residual)>.05)&~foreground
    patch=valid&~foreground&~inconsistent
    patch=cv2.erode(patch.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    errors=np.abs(residual[patch])
    return dict(board_patch_pixels=int(patch.sum()),foreground_pixels=int(foreground.sum()),
        inconsistent_pixels=int(inconsistent.sum()),invalid_pixels=int((~data['valid_depth']).sum()),
        max_abs_residual_m=float(errors.max()) if len(errors) else None,
        p99_abs_residual_m=float(np.quantile(errors,.99)) if len(errors) else None,
        scope='eroded measured board patches; foreground excluded; no carton truth; not full sensor validation')


def margin_audit_accepted(frames,margin):
    """Require actual board samples on all three cameras before low-margin use."""
    formal={'scene_camera','left_wrist_camera','right_wrist_camera'}
    return (set(frames)==formal and all(
        frame['board_patch_pixels']>=500 and frame['max_abs_residual_m'] is not None and
        frame['max_abs_residual_m']<=margin/2 for frame in frames.values()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run_dir',type=Path)
    parser.add_argument('--prefix',default='pregrasp');args=parser.parse_args()
    config=json.loads((args.run_dir/'config_used.json').read_text(encoding='utf-8'))
    results={}
    for name in ('scene_camera','left_wrist_camera','right_wrist_camera'):
        path=args.run_dir/(args.prefix+'_'+name+'.npz')
        if not path.exists():continue
        with np.load(path) as data:results[path.stem]=audit_frame(data,depth_backboards(config))
    report=dict(scope='offline static planar calibration residuals only',frames=results,
        fixed_surface_source='station_priors.depth_backboards; exact experiment config',
        limitations=['no calibrated real-depth noise model','no strict exposure-ID verification',
                     'excluded foreground pixels do not become free space'])
    output=args.run_dir/(args.prefix+'_depth_error_audit.json')
    output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
