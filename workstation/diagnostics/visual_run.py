"""Simulation runner and independent privileged evaluator, outside online skill."""
import json
import hashlib
from pathlib import Path
from workstation.perception.scene_estimate import ScenePriors
from workstation.simulation.robot_driver import from_environment
from workstation.skills.visual_pick_place import VisualPickPlace
from workstation.diagnostics.vision_check import save_observation
from workstation.task_logic import slot_errors


def run(env, segmenter=None,max_items=None):
    root=Path(__file__).resolve().parents[2]
    sources=[root/'workstation/app.py',root/'workstation/task_logic.py',
             root/'workstation/diagnostics/depth_error_audit.py',
             *(root/'workstation/simulation'/name for name in
               ('robot_driver.py','environment.py','visual_fixtures.py','station_priors.py'))]
    for directory in ('observations','perception','planning','skills','models'):
        sources.extend((root/'workstation'/directory).glob('*.py'))
    env._write_json('online_source_hashes.json',{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sources})
    import zipfile
    with zipfile.ZipFile(env.output/'online_source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for source in sources:archive.write(source,str(source.relative_to(root)))
    env.safe_to_index=False
    for _ in range(round(3./env.c['physics']['dt'])):
        env.step(monitor_conveyor=False,render=False)
    robot=from_environment(env)
    env._write_json('robot_model_manifest.json',{name:dict(sha256=model.sha256,
        lower_m=model.lower.tolist(),upper_m=model.upper.tolist()) for name,model in robot.collision_links.items()})
    reads=0
    def read_packet():
        nonlocal reads
        if not reads:
            packet=env.get_observation_packet(refresh=True)
        else:
            # Physics remains frozen through all three render passes; each pass
            # uses the existing camera system's matched robot-state latch.
            for _ in range(3): env.camera_system.render()
            packet=env.get_observation_packet(refresh=False)
        reads+=1
        return packet
    def trace_scene(packet,scene):
        # Encoder/visual diagnostics only. Never feed a privileged record back.
        record=dict(scene=scene.to_dict(),robot_state={name:dict(time_s=state.sample_time_s,
            qpos=state.qpos.tolist(),tcp_fk_m=robot.fk(name,state.joint_positions_rad)[0].tolist(),
            commanded_qpos=robot.targets[name].tolist()) for name,state in packet.robot_state.items()})
        with (env.output/'visual_trace.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
    margin=float(env.c.get('diagnostic_optical_depth_error_margin_m',.003))
    if margin<.003:
        from workstation.diagnostics.depth_error_audit import audit_frame,margin_audit_accepted
        from workstation.simulation.station_priors import depth_backboards
        packet=read_packet();audits={}
        for name,frame in packet.cameras.items():
            if frame is None or packet.camera_status[name]!='OK':continue
            audits[name]=audit_frame(dict(K=frame.K,T_workcell_from_camera_cv=frame.T_workcell_from_camera_cv,
                depth_m=frame.depth_m,valid_depth=frame.valid_depth),depth_backboards(env.c))
        gate=dict(accepted=margin_audit_accepted(audits,margin),optical_depth_error_margin_m=margin,
            frames=audits,scope='controlled static-board initial calibration; foreground not marked free',
            criterion='each formal camera >=500 eroded plane samples; max residual <= half selected margin')
        env._write_json('optical_depth_calibration_gate.json',gate)
        if not gate['accepted']:
            env._write_json('visual_task_report.json',dict(success=False,scope='calibration_before_actor',
                failure_reason='OPTICAL_DEPTH_CALIBRATION_FAILED',events=[],held_on_exit=False,
                max_visual_lift_m=0.,calibration_gate=gate))
            raise RuntimeError('Optical depth margin rejected by current three-camera calibration')
    args=(robot,read_packet,ScenePriors.from_config(env.c),env.c['conveyor']['slots_xy'],
          lambda prefix,packet,scene:save_observation(env.output,prefix,packet,scene),trace_scene,segmenter)
    if max_items is None:skill=VisualPickPlace(*args)
    else:
        from workstation.skills.visual_sorting import VisualSortingTask
        skill=VisualSortingTask(*args,max_items=max_items)
    result=skill.run()
    result['observation_refresh']=dict(initial_render_passes=env.camera_system.settings['warmup_render_frames'],
        subsequent_render_passes=3,physics_frozen_during_refresh=True,
        strict_annotator_exposure_ids_verified=False,reads=reads)
    env._write_json('visual_task_report.json',result)
    # AFTER actor has returned. Never feed these states or errors back to the skill.
    evaluation=dict(scope='offline_final_pose_only',actor_success=result['success'],
        states=[b.to_dict() for b in env.observe_ground_truth()],
        slot_errors={b.name:[slot_errors(b,xy,env.c) for xy in env.c['conveyor']['slots_xy']]
                     for b in env.observe_ground_truth()})
    env._write_json('privileged_evaluation.json',evaluation)
    if not result['success']:
        raise RuntimeError('Visual skill failed: '+result.get('failure_reason',result.get('stop_reason','unknown')))
    return result
