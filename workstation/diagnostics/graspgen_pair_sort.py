"""Two visual pair cycles in one unchanged four-carton physics scene."""
from copy import copy
from time import perf_counter
import numpy as np

from workstation.diagnostics.graspgen_pair import check_pair,source_cartons
from workstation.diagnostics.vision_check import save_observation
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors
from workstation.planning.pair_task import VisualConveyorIndex,choose_source_pair,match_stationary_sources
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
    from workstation.simulation.robot_driver import from_environment
    root=env.output;started=perf_counter()
    report=dict(success=False,scope='two_pairs_four_initial_cartons_live_model_and_visual_conveyor',rounds=[],indexing=[])
    env.safe_to_index=False
    env.conveyor=VisualConveyorIndex(env.c)
    robot=from_environment(env,paired_armature_floor=.01)
    priors=ScenePriors.from_config(env.c)
    def new_estimator():
        return CartonEstimator(priors,robot_point_filter=robot.observed_robot_points,
                               robot_filter_factory=robot.prepare_observed_robot_filter)
    estimator=new_estimator()
    def observe():
        packet=env.get_observation_packet(refresh=True)
        if any(packet.camera_status.get(n)!='OK' for n in ('scene_camera','left_wrist_camera','right_wrist_camera')):
            raise RuntimeError('PAIR_TASK_CAMERA_UNAVAILABLE')
        scene=estimator.estimate(packet);robot.set_observation(packet,scene,priors)
        return packet,scene
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
                packet,scene=observe();next_check=env.conveyor.elapsed+.2
                views=[robot.observed_workspace.slot(xy) for xy in env.c['conveyor']['slots_xy']]
                arms_clear=empty_arms_clear(robot,packet,env.c)
                checks.append(dict(time_s=packet.observation_time_s,arms_clear=arms_clear,slot_views=views))
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
    try:
        for _ in range(round(3/robot.dt)):env.step(monitor_conveyor=False,render=False)
        observe();robot.hold(.2);packet,scene=observe()
        originals=source_cartons(scene,env.c['table'])
        if len(originals)!=4:raise RuntimeError('PAIR_TASK_REQUIRES_FOUR_OBSERVED_SOURCES')
        remaining={o.track_id:dict(position_m=o.position_m,size_m=o.size_m,axis_yaw_rad=o.axis_yaw_rad) for o in originals}
        report['initial_visual_sources']=remaining.copy()
        pair_args=copy(args);pair_args.mode='graspgen-pair-place'
        for round_index in (1,2):
            env.output=root/f'round_{round_index}';env.output.mkdir()
            robot.held_visual_geometry=None;robot.stationary_payload_keepout=None
            robot.plan_cache.clear()
            from workstation.planning.occlusion_memory import OcclusionMemory
            robot.occlusion_memory=OcclusionMemory()
            estimator=new_estimator();observe();robot.hold(.2);packet,scene=observe()
            matches=match_stationary_sources(scene,remaining)
            if matches.keys()!=remaining.keys() or len(source_cartons(scene,env.c['table']))!=len(remaining):
                raise RuntimeError('PAIR_TASK_REMAINING_SOURCES_UNCONFIRMED')
            chosen=choose_source_pair(list(matches.values()),robot.base_config)
            selected_keys=[k for k,o in matches.items() if o.track_id in {x.track_id for x in chosen}]
            reserved={k:remaining[k] for k in selected_keys}
            def select(current):
                matches=match_stationary_sources(current,reserved)
                if matches.keys()!=reserved.keys():raise RuntimeError('PAIR_TASK_RESERVED_SOURCE_LOST')
                return list(matches.values())
            save_observation(env.output,'graspgen_source',packet,scene)
            begin=perf_counter()
            result=check_pair(env,pair_args,packet,scene,estimator,robot=robot,target_selector=select)
            report['rounds'].append(dict(round=round_index,success=result['success'],reserved_source_keys=selected_keys,
                wall_s=perf_counter()-begin,action_wall_s=result.get('action_wall_s'),placements=result['placements'],
                transfer_strategy=result.get('transfer_strategy'),model_request_count=len(result['model_requests'])))
            if not result['success']:raise RuntimeError('PAIR_TASK_ROUND_FAILED')
            for key in selected_keys:del remaining[key]
            packet,scene=index_pair(result['placements'],round_index)
            env.output=root;env._write_json('graspgen_pair_sort.json',report)
        if source_cartons(scene,env.c['table']):raise RuntimeError('PAIR_TASK_SOURCES_REMAIN')
        report.update(success=True,verified_placements=4,completed_pairs=2,visually_cleared_batches=env.conveyor.completed_batches)
    except Exception as error:
        report['failure_reason']=str(error)
        raise
    finally:
        env.output=root;report['total_wall_s']=perf_counter()-started
        report['conveyor_events']=env.conveyor.events
        env._write_json('graspgen_pair_sort.json',report)
