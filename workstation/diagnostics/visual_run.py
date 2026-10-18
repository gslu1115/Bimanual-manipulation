"""Simulation runner and independent privileged evaluator, outside online skill."""
import json
import hashlib
from pathlib import Path
from workstation.perception.scene_estimate import ScenePriors
from workstation.simulation.robot_driver import from_environment
from workstation.skills.visual_pick_place import VisualPickPlace
from workstation.diagnostics.vision_check import save_observation
from workstation.task_logic import slot_errors


def run(env, segmenter=None,max_items=None,graspgen=None,side_adjustment=False):
    from workstation.diagnostics.wall_timing import WallTiming
    from workstation.planning.observed_workspace import ObservedWorkspace
    timing=WallTiming()
    timing.wrap(env,'step','physics_step')
    timing.wrap(env.camera_system,'render','camera_render')
    timing.wrap(ObservedWorkspace,'_build_occupied','depth_obstacle_mapping')
    timing.wrap(ObservedWorkspace,'_evaluate_unknown','free_space_mapping')
    try:
        return _run(env,segmenter,max_items,graspgen,side_adjustment,timing)
    finally:
        timing.restore()
        env._write_json('performance.json',timing.report())


def _run(env, segmenter,max_items,graspgen,side_adjustment,timing):
    root=Path(__file__).resolve().parents[2]
    sources=[root/'workstation/app.py',root/'workstation/task_logic.py',
             root/'workstation/diagnostics/depth_error_audit.py',
             *(root/'workstation/simulation'/name for name in
               ('robot_driver.py','environment.py','visual_fixtures.py','station_priors.py','display_refresh.py'))]
    sources.extend((root/'workstation/diagnostics'/name for name in ('visual_run.py','vision_check.py','identity_capture.py')))
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
    for method,label in (('validate_path','path_validation'),('cartesian_knots','cartesian_ik'),
                         ('checked_joint_plan','joint_planning'),('set_observation','workspace_setup')):
        timing.wrap(robot,method,label)
    env._write_json('robot_model_manifest.json',{name:dict(sha256=model.sha256,
        lower_m=model.lower.tolist(),upper_m=model.upper.tolist()) for name,model in robot.collision_links.items()})
    reads=0
    capture_timing={}
    estimation_before_read=0.
    identity_capture=None
    last_timing_write=timing.clock()
    @timing.timed('sensor_read')
    def read_packet():
        nonlocal reads,estimation_before_read
        import time
        start=timing.clock();wall_start=time.time()
        estimation_before_read=timing.rows.get('scene_estimation',{}).get('inclusive_s',0.)
        if not reads:
            packet=env.get_observation_packet(refresh=True)
        else:
            # Physics remains frozen through all three render passes; each pass
            # uses the existing camera system's matched robot-state latch.
            for _ in range(3): env.camera_system.render()
            packet=env.get_observation_packet(refresh=False)
        reads+=1
        capture_timing.update(read_wall_start_s=wall_start,read_monotonic_start_s=start,
                              read_duration_s=timing.clock()-start)
        return packet
    @timing.timed('trace_write')
    def trace_scene(packet,scene):
        nonlocal last_timing_write
        # Encoder/visual diagnostics only. Never feed a privileged record back.
        record=dict(scene=scene.to_dict(),robot_state={name:dict(time_s=state.sample_time_s,
            qpos=state.qpos.tolist(),tcp_fk_m=robot.fk(name,state.joint_positions_rad)[0].tolist(),
            commanded_qpos=robot.targets[name].tolist()) for name,state in packet.robot_state.items()})
        record['robot_self_filter_counts']=dict(getattr(getattr(skill,'estimator',None),'last_robot_filter_counts',{}))
        record['observation_timing']=dict(capture_timing,trace_monotonic_s=timing.clock(),
            estimation_duration_s=timing.rows.get('scene_estimation',{}).get('inclusive_s',0.)-estimation_before_read)
        estimator=getattr(skill,'estimator',None)
        if getattr(estimator,'record_association',False):
            record['association']=estimator.association_events
        if identity_capture is not None:identity_capture.observe(packet,scene,record)
        with (env.output/'visual_trace.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        if timing.clock()-last_timing_write>=10.:
            env._write_json('performance.json',timing.report())
            last_timing_write=timing.clock()
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
    from workstation.planning.task_targets import visual_target_slots
    args=(robot,read_packet,ScenePriors.from_config(env.c),visual_target_slots(env.c),
          timing.timed('save_observation',lambda prefix,packet,scene:save_observation(env.output,prefix,packet,scene)),trace_scene,segmenter)
    grasp_source=None
    if graspgen is not None:
        from workstation.diagnostics.graspgen_calibration import verify
        from workstation.runtime_paths import isaac_root
        from workstation.models.graspgen_proxy import GraspGenProxy
        from workstation.models.graspgen_source import GraspGenSource
        calibration=verify(env,robot,isaac_root(),graspgen.graspgen_assets)
        grasp_source=GraspGenSource(GraspGenProxy(graspgen.graspgen_proxy),calibration,graspgen.num_grasps,
                                   accepted_postures=('SIDE',) if side_adjustment else ('UPRIGHT',))
        timing.wrap(grasp_source,'generate','grasp_generation')
    if env.c.get('grasp_backend')=='centered':
        from workstation.planning.centered_grasp import CenteredGraspSource
        grasp_source=CenteredGraspSource()
        timing.wrap(grasp_source,'generate','grasp_generation')
    if side_adjustment:
        from workstation.skills.visual_side_adjustment import VisualSideAdjustment
        if max_items is not None:raise ValueError('Side adjustment is a controlled single-object skill')
        skill=VisualSideAdjustment(*args,grasp_source=grasp_source)
    elif max_items is None:
        skill_type=VisualPickPlace
        if env.c.get('motion_backend')=='cumotion':
            from workstation.skills.cumotion_pick_place import CuMotionPickPlace
            skill_type=CuMotionPickPlace
        skill=skill_type(*args,grasp_source=grasp_source,
            observation_period_s=env.c.get('visual_observation_period_s') if grasp_source is not None else None)
    else:
        from workstation.skills.visual_sorting import VisualSortingTask
        skill=VisualSortingTask(*args,max_items=max_items)
    if hasattr(skill,'estimator'):timing.wrap(skill.estimator,'estimate','scene_estimation')
    if env.camera_system.settings['save_images'] and hasattr(skill,'estimator'):
        from workstation.diagnostics.identity_capture import IdentityCapture
        skill.estimator.record_association=True
        identity_capture=IdentityCapture(env.output,save_observation)
    if hasattr(skill,'_read_holding_surface'):timing.wrap(skill,'_read_holding_surface','holding_estimation')
    result=skill.run()
    if getattr(robot,'refresh_display',None) is not None:
        result['display_refresh']=robot.refresh_display.report()
    result['observation_refresh']=dict(initial_render_passes=env.camera_system.settings['warmup_render_frames'],
        subsequent_render_passes=3,physics_frozen_during_refresh=True,
        strict_annotator_exposure_ids_verified=False,reads=reads)
    if hasattr(skill,'_monitor_period_steps'):
        result['observation_refresh'].update(monitor_period_steps=skill._monitor_period_steps(),
            monitor_period_sim_s=skill._monitor_period_steps()*robot.dt,
            maximum_frame_age_s=skill.priors.max_age_s,
            motion_feedback_max_age_s=getattr(robot,'visual_feedback_max_age_s',skill.priors.max_age_s),
            inter_observation_pose='encoder FK with last visually supported grasp; not a fresh visual measurement',
            immediate_phase_observations=True)
        if result.get('motion_backend')=='isaac6.1_pick_place_cumotion_rmpflow':
            result['observation_refresh'].update(policy='official_controller_phase_boundaries_and_watchdog',
                watchdog_period_sim_s=skill.watchdog_period_s,
                periodic_observations=None,controller_policy=result['controller_policy'])
        elif getattr(skill,'observation_period_s',None) is not None:
            result['observation_refresh'].update(policy='adaptive_phase_and_motion',
                fast_period_sim_s=min(.1,skill.priors.max_age_s*2/3),
                stable_translation_period_sim_s=skill.observation_period_s,
                empty_motion_period_sim_s=skill.observation_period_s,
                empty_motion_phases=['PREGRASP','RETREAT','HOME'],
                stable_translation_max_rotation_deg=5.,
                periodic_observations=skill.periodic_observations,
                monitor_period_fields_scope='final state only; not a fixed whole-task interval')
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
