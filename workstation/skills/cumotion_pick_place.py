"""Opt-in continuous official controller with existing RGB-D preparation/verification."""
from __future__ import annotations

from dataclasses import asdict
import math
import time
import numpy as np

from workstation.observations.camera_geometry import pose_matrix, quaternion_from_matrix
from workstation.planning.graspgen_candidates import upright_body_rotation
from workstation.planning.selection import nearest_upright_slot_rotation
from workstation.skills.visual_pick_place import VisualPickPlace


def placement_goal(slot, support_z, size, release_gap, body_rotation, offset, relative_rotation):
    desired = nearest_upright_slot_rotation(body_rotation)
    rotation = desired@np.asarray(relative_rotation).T
    centre = np.array([*slot, support_z+size[2]/2+release_gap])
    return centre-rotation@np.asarray(offset), rotation


class CuMotionPickPlace(VisualPickPlace):
    run_scope = 'controlled_upright_single_carton_official_controller_development'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.grasp_source is None:
            raise ValueError('cuMotion development requires the verified GraspGen pose')
        # Heavy perception happens at state boundaries and a bounded watchdog,
        # not in every physics tick. Encoders and proposed commands are checked
        # continuously; intervening payload poses are explicitly predictions.
        self.robot.monitor = None
        self.watchdog_period_s = 2.
        self.estimator.track_retention_s=max(self.estimator.track_retention_s,
                                             self.watchdog_period_s+self.priors.max_age_s)

    def _monitor_period_steps(self):
        return max(1,math.ceil(getattr(self,'watchdog_period_s',2.)/self.robot.dt))

    def execute_selected(self, result, chosen, prepared_q, obj, slot):
        candidate = self.candidate
        self.robot.monitor = None
        pick = np.asarray(candidate.position_m)
        pick_rotation = pose_matrix([0., 0., 0.], candidate.orientation_wxyz)[:3, :3]
        body_rotation = upright_body_rotation(obj)
        offset = pick_rotation.T@(np.asarray(obj.position_m)-pick)
        relative = pick_rotation.T@body_rotation
        place, place_rotation = placement_goal(slot, self.priors.support_z_m,
            obj.size_m, self.release_gap_m, body_rotation, offset, relative)
        # Same bounded position preload magnitude and 30 N physical finger drive.
        close_position = max(0., candidate.closing_width_m/2-.004)
        driver = self.robot.create_pick_place_controller(self.arm, self.scene,
            candidate.target_id, pick, pick_rotation, place, place_rotation, close_position,
            approach_direction=candidate.approach_direction)
        wait=getattr(self.robot,'wait_for_start',None)
        if wait is not None:wait()
        result['motion_backend'] = 'isaac6.1_pick_place_cumotion_rmpflow'
        result['executed_candidate'] = asdict(candidate)
        result['controller_policy'] = dict(control_period_sim_s=2*self.robot.dt,
            perception='phase boundaries plus 2 s simulation-time watchdog; synchronous boundary capture',
            per_command_guard='sampled existing robot/payload collisions and observed obstacles',
            local_unknown_space='fresh phase preflight; no repeated unknown-ray query per control tick',
            between_images='encoder FK and last visually supported grasp relation; prediction, not measurement',
            surface_gripper=False, whole_arm_unknown_space_verified=False,
            continuous_collision_proof=False, closing_position_m=close_position)
        result['controller_policy']['contact_motion']='model-direction approach and fixed-attitude Cartesian contact segments; RMPflow for free travel'
        control_trace=[]
        result['controller_trace']=control_trace
        last_sample=-float('inf')
        def record_step():
            nonlocal last_sample
            state=self.robot.read_states()[self.arm]
            if state.sample_time_s-last_sample<.05-1e-9 or len(control_trace)>=4000:return
            last_sample=state.sample_time_s
            p,r=self.robot.fk(self.arm,state.joint_positions_rad)
            target_p,target_r=self.robot.fk(self.arm,self.robot.targets[self.arm][:7])
            control_trace.append(dict(time_s=state.sample_time_s,phase=driver.phase,
                tcp_m=np.asarray(p).tolist(),orientation_wxyz=quaternion_from_matrix(r).tolist(),
                commanded_tcp_m=np.asarray(target_p).tolist(),commanded_orientation_wxyz=quaternion_from_matrix(target_r).tolist(),
                fingers_m=state.finger_positions_m.tolist(),qpos=state.qpos.tolist(),qvel=state.qvel.tolist()))
        self.robot.record_control_step=record_step
        self.robot.visual_feedback_max_age_s=self.watchdog_period_s+self.priors.max_age_s
        previous_phase = driver.phase
        phase_start = time.perf_counter()
        motion_start = phase_start
        phase_records = []
        result['controller_phases'] = phase_records
        self.event('PREGRASP')
        next_observation = self.watchdog_period_s
        released_centre = None
        drop_bound = None
        try:
            while not driver.done:
                old_phase = driver.phase
                target, velocity = driver.propose(2*self.robot.dt)
                if driver.phase != previous_phase:
                    used_cartesian=False
                    now=time.perf_counter()
                    phase_records.append(dict(phase=previous_phase,wall_s=now-phase_start,
                        end_controller_sim_s=driver.time_s))
                    phase_start=now
                    previous_phase=driver.phase
                    if driver.phase=='DESCEND_PICK':
                        self.observe('pregrasp')
                        current=[o for o in self.scene.objects if o.track_id==candidate.target_id]
                        if len(current)!=1:raise RuntimeError('PREGRASP_TARGET_UNCERTAIN')
                        self.candidate=self._refresh_candidate(candidate,current[0])
                        self.event('APPROACH')
                        self._cartesian_segment(driver,pick,pick_rotation,contact=True,approach=True)
                        self.observe('approached')
                        current=[o for o in self.scene.objects if o.track_id==candidate.target_id]
                        if len(current)!=1:raise RuntimeError('APPROACHED_TARGET_UNCERTAIN')
                        self.candidate=self._refresh_candidate(candidate,current[0])
                        used_cartesian=True
                    elif driver.phase=='GRASP':
                        self.event('CLOSE')
                    elif driver.phase=='LIFT':
                        tcp,grasp_rotation=self.latch_closed_payload(result,True,preload=False)
                        self.event('LIFT')
                        self._cartesian_segment(driver,tcp+[0.,0.,.18],grasp_rotation,contact=True)
                        used_cartesian=True
                    elif driver.phase in ('APPROACH_PLACE','DESCEND_PLACE'):
                        label='lifted' if driver.phase=='APPROACH_PLACE' else 'transferred'
                        self.observe(label)
                        measurement=self.holding_measurement()
                        if measurement is None:raise RuntimeError('CURRENT_HELD_SURFACE_UNSUPPORTED')
                        if self.max_visual_lift < .08:raise RuntimeError('NO_VISUAL_LIFT_PROOF')
                        relation=self.robot.held_visual_geometry
                        place,place_rotation=placement_goal(slot,self.priors.support_z_m,obj.size_m,
                            self.release_gap_m,measurement['rotation'],relation['offset'],relation['relative_rotation'])
                        driver.update_place_from_measurement(place,place_rotation)
                        result['model_placement']=dict(target_tcp_m=place.tolist(),
                            orientation_wxyz=quaternion_from_matrix(place_rotation).tolist(),
                            release_gap_m=self.release_gap_m,observation_time_s=self.scene.observation_time_s,
                            visual_offset_tcp_m=relation['offset'].tolist(),relative_rotation=relation['relative_rotation'].tolist())
                        transfer=driver.phase=='APPROACH_PLACE'
                        self.event('TRANSFER' if transfer else 'LOWER')
                        if transfer:
                            self._phase_path(place+[0.,0.,.18],place_rotation,contact=False)
                        else:
                            self._cartesian_segment(driver,place,place_rotation,contact=True)
                            used_cartesian=True
                    elif driver.phase=='RELEASE':
                        self.observe('lowered')
                        if self.holding_measurement() is None:raise RuntimeError('CURRENT_HELD_SURFACE_UNSUPPORTED')
                        tcp,rotation=self.robot.fk(self.arm,self.packet.robot_state[self.arm].joint_positions_rad)
                        relation=self.robot.held_visual_geometry
                        released_centre=tcp+rotation@relation['offset']
                        released_rotation=rotation@relation['relative_rotation']
                        bottom=released_centre[2]-np.abs(released_rotation[2])@(self.hold_tracker.size/2)
                        drop_bound=max(0.,float(bottom-self.priors.support_z_m))+self.hold_tracker.uncertainty_m
                        if drop_bound>.020:raise RuntimeError('RELEASE_HEIGHT_UNSUPPORTED')
                        result['release_geometry']=dict(expected_lowest_corner_m=float(bottom),
                            observed_supported_body=True,drop_bound_m=drop_bound,
                            source='current RGB-D support and image-time FK')
                        self.event('OPEN');self.releasing=True
                    elif driver.phase=='RETREAT':
                        self.observe_released(result,drop_bound,released_centre)
                        self.event('RETREAT')
                        self._phase_path(place+[0.,0.,.18],place_rotation,contact=True)
                    driver.update_scene(self.scene)
                    next_observation=driver.time_s+self.watchdog_period_s
                    # The pending desired command belongs to the previous phase
                    # and predates the checked Cartesian motion. Never apply it.
                    if used_cartesian:continue
                elif driver.time_s>=next_observation and not driver.done:
                    self.observe('controller_watchdog')
                    if self.held and not self.releasing and self.holding_measurement() is None:
                        raise RuntimeError('CURRENT_HELD_SURFACE_UNSUPPORTED')
                    driver.update_scene(self.scene)
                    next_observation=driver.time_s+self.watchdog_period_s
                if driver.done:
                    break
                measured=self.robot.read_states()[self.arm].joint_positions_rad
                if np.max(np.abs(target[:7]-measured))>.18:
                    raise RuntimeError('CUMOTION_JOINT_TRACKING_ERROR')
                contact=old_phase not in ('APPROACH_PICK','APPROACH_PLACE')
                self.robot.validate_path(self.arm,[measured,target[:7]],self.scene,candidate.target_id,
                    held_size=self.hold_tracker.size if self.held else None,
                    support_contact=contact,target_contact=contact and not self.held,
                    finger_positions=np.maximum(target[7:],self.robot.read_states()[self.arm].finger_positions_m),
                    check_local_unknown=False,check_approach_corridor=False)
                self.robot.targets[self.arm]=target
                self.robot.command_velocities(self.arm,velocity)
                for _ in range(2):self.robot.tick()
            self.event('HOME')
            self.robot.command_velocities(self.arm,np.zeros(9))
            self.robot.execute_joint(self.arm,self.robot.home,self.scene,candidate.target_id)
        finally:
            self.robot.record_control_step=None
            self.robot.command_velocities(self.arm,np.zeros(9))
            result['controller_timing']=dict(motion_through_home_wall_s=time.perf_counter()-motion_start,
                controller_sim_s=driver.time_s,controller_and_state_wall_s=driver.controller_wall_s,
                controller_calls=driver.calls,phase_at_exit=driver.phase)

    def _cartesian_segment(self,driver,target,rotation,contact,approach=False):
        self.robot.command_velocities(self.arm,np.zeros(9))
        start=self.robot.read_states()[self.arm].sample_time_s
        self.robot.monitor=self.monitor
        try:
            self.robot.execute_cartesian(self.arm,np.asarray(target),0.,0.,self.scene,
                self.candidate.target_id,speed=.10,
                held_size=self.hold_tracker.size if self.held else None,
                support_contact=contact,check_approach_corridor=approach,
                orientation_wxyz=quaternion_from_matrix(rotation))
        finally:self.robot.monitor=None
        elapsed=self.robot.read_states()[self.arm].sample_time_s-start
        driver.resync_after_cartesian(elapsed)

    def _phase_path(self,target,rotation,contact,approach=False):
        """One fresh preflight per segment; each actual RMP command is also guarded."""
        state=self.robot.read_states()[self.arm]
        tcp,current_r=self.robot.fk(self.arm,state.joint_positions_rad)
        path=self.robot.cartesian_knots(self.arm,tcp,target,0.,0.,.15,state.joint_positions_rad,
            orientation0=quaternion_from_matrix(current_r),orientation1=quaternion_from_matrix(rotation))
        self.robot.validate_path(self.arm,path,self.scene,self.candidate.target_id,
            held_size=self.hold_tracker.size if self.held else None,support_contact=contact,
            target_contact=contact and not self.held,check_approach_corridor=approach)
