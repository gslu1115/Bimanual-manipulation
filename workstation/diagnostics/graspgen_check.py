"""Snapshot candidate evaluation; deliberately no robot execution or TCP claim."""
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from workstation.models.graspgen_client import visual_clouds
from workstation.models.graspgen_proxy import GraspGenProxy
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors
from workstation.diagnostics.vision_check import save_observation
from workstation.observations.camera_geometry import quaternion_from_matrix

def proposal_records(result,target_id,sequence_ids):
    records=[]
    for i,(pose,score) in enumerate(zip(result['base_poses_workcell'],result['scores'])):
        records.append(dict(candidate_id=result['request_id']+':'+str(i),source='graspgen',target_id=target_id,
            observation_time_s=result['observation_time_s'],sequence_ids=sequence_ids,frame='workcell',
            T_workcell_from_graspgen_base=pose.tolist(),position_m=pose[:3,3].tolist(),
            orientation_wxyz=quaternion_from_matrix(pose[:3,:3]).tolist(),
            approach_axis_workcell=pose[:3,2].tolist(),closing_axis_workcell=pose[:3,0].tolist(),
            model_score=float(score),score_is_success_probability=False,
            opening_m=None,opening_source='not provided by official service',project_tcp_pose=None,
            executable=False,rejection_reasons=['TCP_CALIBRATION_UNVERIFIED','OPENING_UNVERIFIED','PATH_NOT_CHECKED'],
            model_metadata=result['metadata']))
    return records

def show_axes(env,records):
    from pxr import UsdGeom,Gf,Vt
    # Original scene, camera installation and collision geometry are untouched.
    for i,record in enumerate(records[:10]):
        matrix=np.asarray(record['T_workcell_from_graspgen_base'])
        for axis,color in enumerate(((1.,.1,.1),(.1,1.,.1),(.1,.4,1.))):
            curve=UsdGeom.BasisCurves.Define(env.stage,f'/GraspGenPreview/g{i}/axis{axis}')
            points=[matrix[:3,3],matrix[:3,3]+matrix[:3,axis]*.04]
            curve.CreateTypeAttr('linear');curve.CreateCurveVertexCountsAttr([2])
            curve.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float,p)) for p in points]))
            curve.CreateWidthsAttr([.0015]);curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])

def check(env,args):
    env.safe_to_index=False
    for _ in range(round(3/env.c['physics']['dt'])):env.step(monitor_conveyor=False,render=False)
    packet=env.get_observation_packet(refresh=True)
    scene=CartonEstimator(ScenePriors.from_config(env.c)).estimate(packet)
    save_observation(env.output,'graspgen_source',packet,scene)
    targets=[o for o in scene.objects if not o.failures and o.visibility=='COMPLETE_TOP']
    if not targets:raise RuntimeError('No unambiguous visually observed target for GraspGen diagnostic')
    target=targets[0]
    clouds=visual_clouds(packet,scene,target.track_id)
    np.savez_compressed(env.output/'graspgen_cloud.npz',target_m=clouds.target_m,scene_m=clouds.scene_m,time_s=clouds.time_s)
    # A saved snapshot experiment, NOT an online latency/freshness acceptance.
    env.timeline.pause()
    proxy=GraspGenProxy(args.graspgen_proxy)
    runs=[];records=[]
    for trial in range(3):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=pool.submit(proxy.infer,clouds,args.num_grasps)
            while not pending.done():
                env.app.update()
                time.sleep(1/60)
            result=pending.result()
        records=proposal_records(result,target.track_id,{n:f.sequence_id for n,f in packet.cameras.items() if f})
        runs.append(dict(trial=trial,round_trip_s=result['round_trip_s'],timing=result['timing'],candidates=records))
    from workstation.planning.selection import assess
    from workstation.simulation.robot_driver import from_environment
    robot=from_environment(env);robot.set_observation(packet,scene,ScenePriors.from_config(env.c))
    geometry=assess(scene,robot.read_states(),robot.feasibility,robot.home)
    report=dict(scope='saved_visual_snapshot_candidates_only',success=all(r['candidates'] for r in runs),
        task_execution=False,TCP_calibration_verified=False,online_freshness_verified=False,
        trials_are_independent=False,source_points=len(clouds.target_m),model_runs=runs,
        geometry_candidates=[asdict(x) for x in geometry],
        comparison_scope='same snapshot; geometry approach checks only; model TCP/path not yet checked')
    env._write_json('graspgen_report.json',report)
    show_axes(env,records)
    import omni.ui as ui
    window=ui.Window('GraspGen candidate preview - no robot execution',width=550,height=170)
    with window.frame:
        with ui.VStack():
            ui.Label(f'Panda GraspGen: {len(records)} candidates from {len(clouds.target_m)} observed points',word_wrap=True)
            ui.Label('RGB axes show model gripper BASE frames, not calibrated robot TCPs.',word_wrap=True)
            ui.Label('Snapshot only. TCP calibration and physical execution are pending.',word_wrap=True)
    env._graspgen_preview_window=window
    print('[graspgen-check] '+str(env.output/'graspgen_report.json'),flush=True)
    if not report['success']:raise RuntimeError('GraspGen returned no candidates')
    return report
