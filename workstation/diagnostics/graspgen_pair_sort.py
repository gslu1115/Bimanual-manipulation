"""Two visual pair cycles in one unchanged four-carton physics scene."""
from copy import copy
from time import perf_counter
import numpy as np

from workstation.diagnostics.graspgen_pair import check_pair,source_cartons,PairAttemptError
from workstation.diagnostics.vision_check import save_observation
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors
from workstation.planning.pair_task import VisualConveyorIndex,match_task_sources
from workstation.planning.joint_pair_selection import JointPairSelection,recoverable_empty_failure
from workstation.planning.collision import AABB,sphere_box_clearance


def empty_arms_clear(robot,packet,config):
    """Image-time encoder FK, open fingers and clearance of the whole loading zone."""
    slots=np.asarray(config['conveyor']['slots_xy']);lo=slots.min(axis=0)-[.12,.15];hi=slots.max(axis=0)+[.12,.15]
    zone=AABB('loading_zone',np.r_[(lo+hi)/2,config['conveyor']['top_z']+.25],np.r_[hi-lo,.5])
    for arm,state in packet.robot_state.items():
        if (abs(state.sample_time_s-packet.observation_time_s)>.02 or
                state.joint_velocities_rad_s is None or
                np.max(np.abs(state.joint_positions_rad-robot.home[:7]))>.03 or
                np.min(state.finger_positions_m)<.035 or np.max(np.abs(state.joint_velocities_rad_s))>.05):return False
        if any(sphere_box_clearance(c,r,zone)<.008 for _,c,r in robot.spheres(arm,state.joint_positions_rad)):return False
    return len(packet.robot_state)==2


def run(env,args):
    from workstation.diagnostics.wall_timing import WallTiming
    from workstation.planning.observed_workspace import ObservedWorkspace
    from workstation.simulation.robot_driver import RobotMotion
    from workstation.perception.held_surface import CuboidHoldingTracker,HeldSurfaceTracker
    from workstation.diagnostics import graspgen_pair,graspgen_pair_place,graspgen_calibration
    from workstation.models import graspgen_client
    from workstation.simulation import robot_driver
    timing=WallTiming();root=env.output
    for obj,name,label in (
        (env,'step','physics_step'),(env.camera_system,'render','camera_render'),
        (env,'_physics_step','physics_engine'),(env,'_app_update','simulation_app_update'),
        (env,'get_observation_packet','sensor_read'),(env,'_write_json','json_write'),
        (env,'_encode_json','json_encode'),(env,'_replace_json','json_file_replace'),
        (env,'observe_robot_states','encoder_acquisition'),
        (CartonEstimator,'estimate','scene_estimation'),
        (CartonEstimator,'resolve_action_sources','source_pose_resolution'),
        (ObservedWorkspace,'_compute_occupied','depth_obstacle_mapping'),
        (ObservedWorkspace,'_compute_fixed_points','depth_fixed_filter'),
        (ObservedWorkspace,'_evaluate_unknown','free_space_mapping'),
        (ObservedWorkspace,'slot','slot_query'),
        (ObservedWorkspace,'_query_slot','slot_query_compute'),
        (RobotMotion,'fk','fk_query'),(RobotMotion,'_compute_fk','fk_compute'),
        (RobotMotion,'chain_poses','chain_geometry_query'),(RobotMotion,'_compute_chain','chain_fk_compute'),
        (RobotMotion,'_path_context','validation_context_query'),
        (RobotMotion,'_prepare_path_context','validation_context_prepare'),
        (RobotMotion,'_validate_observed','validation_observed_obstacles'),
        (RobotMotion,'_validate_target_bodies','validation_target_body'),
        (RobotMotion,'_validate_payload','validation_payload'),
        (RobotMotion,'spheres','sphere_query'),(RobotMotion,'_compute_spheres','sphere_compute'),
        (robot_driver,'check_spheres','collision_math'),
        (graspgen_pair,'check_spheres','collision_math'),
        (graspgen_pair_place,'check_spheres','collision_math'),
        (graspgen_pair_place,'sphere_box_clearances','payload_collision_math'),
        (RobotMotion,'solve_pose','ik_solve'),
        (RobotMotion,'validate_path','path_validation'),
        (RobotMotion,'cartesian_knots','cartesian_ik'),
        (RobotMotion,'checked_joint_plan','joint_planning'),
        (RobotMotion,'pose_feasibility','candidate_feasibility'),
        (RobotMotion,'set_observation','workspace_setup'),
        (RobotMotion,'hold','hold'),(RobotMotion,'execute_joint','joint_execution'),
        (CuboidHoldingTracker,'observe','held_pose_estimation'),
        (HeldSurfaceTracker,'collision_estimate','released_pose_estimation'),
        (graspgen_client,'infer_rotated_view','model_request'),
        (graspgen_calibration,'verify','asset_calibration'),
        (graspgen_pair,'assess','candidate_assessment'),
        (graspgen_pair,'paired_pregrasp','pair_selection'),
        (graspgen_pair,'drive_empty_pregrasps','pregrasp_execution'),
        (graspgen_pair_place,'drive_empty_pregrasps','phase_execution'),
        (graspgen_pair_place,'transfer_schedule','transfer_schedule'),
        (graspgen_pair_place,'execute_parallel_home','parallel_home')):
        timing.wrap(obj,name,label)
    try:
        return _run(env,args,timing)
    finally:
        timing.restore();env.output=root
        env._write_json('performance.json',timing.report())


def _run(env,args,timing):
    from workstation.simulation.robot_driver import from_environment
    root=env.output;started=perf_counter()
    report=dict(success=False,scope='two_pairs_four_initial_cartons_live_model_and_visual_conveyor',rounds=[],indexing=[],attempts=[])
    env.safe_to_index=False
    env.conveyor=VisualConveyorIndex(env.c)
    robot=from_environment(env,paired_armature_floor=.01)
    robot.wall_timing=timing
    if getattr(args,'pair_collision_backend','scalar')=='curobo':
        from workstation.planning.curobo_pair import PairPathClient
        robot.pair_path_client=PairPathClient(args.graspgen_proxy)
    for method,label in (('command','joint_command'),('command_velocities','velocity_command'),
                         ('read_states','encoder_read')):
        timing.wrap(robot,method,label)
    priors=ScenePriors.from_config(env.c)
    def new_estimator():
        return CartonEstimator(priors,robot_point_filter=robot.observed_robot_points,
                               robot_filter_factory=robot.prepare_observed_robot_filter,persistent_identities=True,
                               known_static=robot.static)
    estimator=new_estimator()
    def observe():
        packet=env.get_observation_packet(refresh=True)
        if any(packet.camera_status.get(n)!='OK' for n in ('scene_camera','left_wrist_camera','right_wrist_camera')):
            raise RuntimeError('PAIR_TASK_CAMERA_UNAVAILABLE')
        scene=estimator.estimate(packet)
        if source_query:scene=estimator.resolve_action_sources(packet,scene,source_query)
        robot.set_observation(packet,scene,priors)
        return packet,scene
    @timing.timed('conveyor_depth_observation')
    def observe_slots():
        from workstation.planning.observed_workspace import ObservedWorkspace
        packet=env.get_observation_packet(refresh=True)
        if any(packet.camera_status.get(n)!='OK' for n in ('scene_camera','left_wrist_camera','right_wrist_camera')):
            raise RuntimeError('PAIR_TASK_CAMERA_UNAVAILABLE')
        workspace=ObservedWorkspace.for_slot_observation(packet,priors,robot.static)
        return packet,[workspace.slot(xy) for xy in env.c['conveyor']['slots_xy']]
    def index_pair(placements,round_index):
        robot.hold(.3);packet,scene=observe()
        ids=[f'round_{round_index}/{p["observed_track"]}' for p in placements.values()]
        env.conveyor.start(ids,empty_arms_clear(robot,packet,env.c))
        env.release_conveyor_interlock();checks=[];next_check=0.
        began=perf_counter()
        try:
            while env.conveyor.state=='INDEXING':
                speed=env.conveyor.advance(robot.dt)
                env.step(monitor_conveyor=False,render=False,conveyor_velocity_m_s=speed)
                if env.conveyor.elapsed+1e-9<next_check:continue
                packet,views=observe_slots();next_check=env.conveyor.elapsed+.2
                arms_clear=empty_arms_clear(robot,packet,env.c)
                checks.append(dict(time_s=packet.observation_time_s,arms_clear=arms_clear,slot_views=views,
                    perception_scope='current three-camera depth occupancy; no whole-table recognition'))
                env.conveyor.observe(packet.observation_time_s,views,arms_clear)
        except Exception:
            env.conveyor.state='FAULT'
            raise
        finally:
            env.step(monitor_conveyor=False,render=False,conveyor_velocity_m_s=0.)
            env.safe_to_index=False
            report['indexing'].append(dict(round=round_index,success=env.conveyor.state=='WAITING',
                wall_s=perf_counter()-began,elapsed_sim_s=env.conveyor.elapsed,checks=checks))
        # Motion has stopped. Demand a NEW settled free view before any reservation.
        robot.hold(.6);packet,scene=observe()
        views=[robot.observed_workspace.slot(xy) for xy in env.c['conveyor']['slots_xy']]
        if not all(v['free'] for v in views):raise RuntimeError('PAIR_TASK_SLOTS_NOT_CLEAR_AFTER_STOP')
        report['indexing'][-1]['stopped_slot_views']=views
        return packet,scene
    def recover_empty(failed,remaining):
        recovery=dict(success=False,scope='checked empty retreat before descent or closure',steps=[])
        began=perf_counter()
        try:
            if not recoverable_empty_failure(failed):raise RuntimeError('PAIR_RECOVERY_PHASE_UNSUPPORTED')
            robot.hold(.15);packet,scene=observe()
            if any(np.min(s.finger_positions_m)<.035 for s in packet.robot_state.values()):
                raise RuntimeError('PAIR_RECOVERY_FINGERS_NOT_OPEN')
            save_observation(env.output,'recovery_start',packet,scene)
            culprit=(failed.get('perception_failure') or failed.get('path_failure',{})).get('arm')
            order=sorted(packet.robot_state,key=lambda a:a!=culprit)
            # First use the existing view. If blocked, try a short checked
            # observation lift, then HOME only if useful sources remain hidden.
            matches=match_task_sources(scene,remaining)
            if len(matches)<2 or failed.get('next_action')=='OBSERVE':
                for arm in order:
                    packet,scene=observe()
                    try:
                        robot.execute_observation_lift(arm,.06,scene)
                    except RuntimeError as error:
                        if robot.motion_started:raise
                        recovery['steps'].append(dict(arm=arm,kind='view_lift_rejected',reason=str(error)))
                        continue
                    recovery['steps'].append(dict(arm=arm,kind='view_lift',plan_check=robot.last_plan_check.copy()))
                    packet,scene=observe();matches=match_task_sources(scene,remaining)
                    if len(matches)>=2:break
            robot.hold(.1);packet,scene=observe();matches=match_task_sources(scene,remaining)
            if len(matches)<2:
                for arm in order:
                    packet,scene=observe()
                    robot.execute_joint(arm,robot.home,scene,None)
                    recovery['steps'].append(dict(arm=arm,kind='home',plan_check=robot.last_plan_check.copy()))
                robot.hold(.2);observe();robot.hold(.1);packet,scene=observe()
                matches=match_task_sources(scene,remaining)
            if (len(matches)<2 or any(o.velocity_m_s is None or
                    np.linalg.norm(o.velocity_m_s)>.015 for o in matches.values()) or
                    any(np.min(s.finger_positions_m)<.035 or s.joint_velocities_rad_s is None or
                        np.max(np.abs(s.joint_velocities_rad_s))>.05 for s in packet.robot_state.values())):
                raise RuntimeError('PAIR_RECOVERY_NOT_STABLE')
            for c in failed['pair']['selection']:
                estimator.mark_identity(c['metadata']['source_visual_object'].get('identity_id'),'available')
            save_observation(env.output,'recovery_ready',packet,scene)
            recovery.update(success=True,observation_time_s=packet.observation_time_s,
                source_tracks={k:o.track_id for k,o in matches.items()})
        except Exception as error:
            recovery['failure_reason']=str(error);raise
        finally:
            recovery['wall_s']=perf_counter()-began
            env._write_json('empty_recovery.json',recovery)
        return recovery
    source_query=()
    try:
        for _ in range(round(3/robot.dt)):env.step(monitor_conveyor=False,render=False)
        observe();robot.hold(.2);packet,scene=observe()
        originals=source_cartons(scene,env.c['table'])
        if len(originals)!=4:raise RuntimeError('PAIR_TASK_REQUIRES_FOUR_OBSERVED_SOURCES')
        remaining={o.track_id:dict(position_m=o.position_m,size_m=o.size_m,axis_yaw_rad=o.axis_yaw_rad) for o in originals}
        estimator.set_task_identities(remaining)
        report['initial_visual_sources']=remaining.copy()
        selection=JointPairSelection(remaining)
        pair_args=copy(args);pair_args.mode='graspgen-pair-place'
        for round_index in (1,2):
            begin=perf_counter()
            round_output=root/f'round_{round_index}';round_output.mkdir()
            for attempt in range(1,5):
                source_query=tuple(remaining)
                env.output=round_output/f'attempt_{attempt}';env.output.mkdir()
                robot.held_visual_geometry=None;robot.stationary_payload_keepout=None
                robot.plan_cache.clear()
                from workstation.planning.occlusion_memory import OcclusionMemory
                robot.occlusion_memory=OcclusionMemory()
                if attempt==1:
                    # The settled registration/index-stop packet is the first velocity sample.
                    # Reattach its map to this attempt's new occlusion memory.
                    first_scene=estimator.resolve_action_sources(packet,scene,source_query)
                    robot.set_observation(packet,first_scene,priors)
                else:
                    observe()
                robot.hold(.2);packet,scene=observe()
                matches=match_task_sources(scene,remaining)
                if len(matches)<2:
                    raise RuntimeError('PAIR_TASK_REMAINING_SOURCES_UNCONFIRMED')
                def select(current):
                    matches=match_task_sources(current,remaining)
                    if len(matches)<2:raise RuntimeError('PAIR_TASK_RESERVED_SOURCE_LOST')
                    return list(matches.values())
                save_observation(env.output,'graspgen_source',packet,scene)
                entry=dict(round=round_index,attempt=attempt,success=False)
                report['attempts'].append(entry)
                try:
                    source_query=()
                    result=check_pair(env,pair_args,packet,scene,estimator,robot=robot,
                                      target_selector=select,selection_policy=selection)
                except PairAttemptError as error:
                    failed=error.report
                    entry.update(failure_reason=str(error),action_wall_s=failed.get('action_wall_s'),
                                 joint_choice=failed.get('joint_choice'))
                    if not recoverable_empty_failure(failed) or attempt==4:raise
                    source_query=tuple(remaining)
                    entry['recovery']=recover_empty(failed,remaining)
                    entry['failure_memory']=selection.record_failure(failed)
                    continue
                entry.update(success=True,joint_choice=result['joint_choice'],action_wall_s=result.get('action_wall_s'))
                selected_keys=result['joint_choice']['source_keys']
                break
            env.output=round_output;env._write_json('graspgen_pair_check.json',result)
            report['rounds'].append(dict(round=round_index,success=result['success'],reserved_source_keys=selected_keys,
                wall_s=perf_counter()-begin,action_wall_s=result.get('action_wall_s'),placements=result['placements'],
                transfer_strategy=result.get('transfer_strategy'),model_request_count=len(result['model_requests'])))
            if not result['success']:raise RuntimeError('PAIR_TASK_ROUND_FAILED')
            for key in selected_keys:del remaining[key]
            packet,scene=index_pair(result['placements'],round_index)
            for key in selected_keys:estimator.mark_identity(key,'delivered')
            report['object_identities']=[vars(v) for v in estimator._identities.values()]
            env.output=root;env._write_json('graspgen_pair_sort.json',report)
        if source_cartons(scene,env.c['table']):raise RuntimeError('PAIR_TASK_SOURCES_REMAIN')
        report.update(success=True,verified_placements=4,completed_pairs=2,visually_cleared_batches=env.conveyor.completed_batches)
    except Exception as error:
        report['failure_reason']=str(error)
        raise
    finally:
        env.output=root;report['total_wall_s']=perf_counter()-started
        if hasattr(robot,'pair_path_client'):
            client=robot.pair_path_client
            report['curobo_pregrasp']=dict(batches=client.batches,samples=client.samples,round_trip_s=client.wall_s)
        report['conveyor_events']=env.conveyor.events
        env._write_json('graspgen_pair_sort.json',report)
