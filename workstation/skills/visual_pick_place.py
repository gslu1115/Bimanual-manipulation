"""Controlled upright visual skill. Limited templates; every phase is reobserved.

Migrates baseline quintic/IK knot/physical gripper phase experience. Does NOT
inherit its truth reads, physical-hand pose reads or truth success predicates.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import math
import numpy as np
from workstation.perception.rgbd_cartons import CartonEstimator, axis_error
from workstation.planning.selection import (assess, select_skill, upright_grasp_position,
    nearest_aligned_yaw, nearest_upright_slot_rotation, UPRIGHT_RELEASE_GAP_M,
    GRASPGEN_UPRIGHT_RELEASE_GAP_M)
from workstation.skills.reobserve import Reobserve, ActiveReobserve, GuidedReobserve
from workstation.perception.held_surface import HeldSurfaceTracker,CuboidHoldingTracker
from workstation.perception.scene_estimate import PoseHypothesis,pose_is_observed
from workstation.observations.camera_geometry import quaternion_from_matrix


class VisualPickPlace:
    accepted_postures=('UPRIGHT',)
    skill_name='DIRECT_PICK_PLACE'
    run_scope='controlled_upright_single_carton'
    release_gap_m=UPRIGHT_RELEASE_GAP_M

    def __init__(self, robot, read_packet, priors, slots_xy, save_scene=None,trace_scene=None,segmenter=None,
                 estimator=None,failure_history=None,eligible=None,grasp_source=None,observation_period_s=None):
        self.robot,self.read_packet,self.priors=robot,read_packet,priors
        self.observation_period_s=None
        if grasp_source is not None and self.accepted_postures==('UPRIGHT',):
            period=1. if observation_period_s is None else float(observation_period_s)
            if not np.isfinite(period) or not .1<=period<=2.:
                raise ValueError('Upright observation period must be 0.1..2 seconds')
            self.observation_period_s=period
        elif observation_period_s is not None:
            raise ValueError('Observation interval override requires upright GraspGen')
        # Receiving an old frame is different from continuing between planned
        # captures. Image freshness stays strict; motion uses encoder prediction.
        self.robot.visual_feedback_max_age_s=(priors.max_age_s if
            self.observation_period_s is None or self.observation_period_s<=priors.max_age_s else
            max(priors.max_age_s,self.observation_period_s+priors.max_age_s+4*getattr(robot,'dt',1/120)))
        self.slots=tuple(tuple(xy) for xy in slots_xy)
        self.estimator=estimator or CartonEstimator(priors,segmenter,
            robot_point_filter=getattr(robot,'observed_robot_points',None),
            robot_filter_factory=getattr(robot,'prepare_observed_robot_filter',None),
            track_retention_s=max(.6,(self.observation_period_s or 0.)+priors.max_age_s))
        self.failure_history=failure_history or {};self.eligible=eligible
        self.save_scene=save_scene or (lambda prefix,packet,scene:None)
        self.trace_scene=trace_scene or (lambda packet,scene:None)
        self.scene=None; self.packet=None; self.events=[]; self.arm=None; self.held=False
        self._monitor_ticks=0; self._missing=0; self.max_visual_lift=0.
        self.initial_position=None; self.candidate=None
        self.hold_tracker=None
        self._hold_observation=None
        self.robot.held_visual_geometry=None
        self.observer_arms=set();self.slot_xy=None
        self.releasing=False
        self.grasp_source=grasp_source
        if grasp_source is not None and self.accepted_postures==('UPRIGHT',):
            self.release_gap_m=GRASPGEN_UPRIGHT_RELEASE_GAP_M
        self.release_tracker=None
        self.observation_anchors=[]
        self.robot.monitor=self.monitor

    def _check_task(self,candidate,obj,scene,slot,**kwargs):
        if self.release_gap_m!=UPRIGHT_RELEASE_GAP_M:kwargs['release_gap_m']=self.release_gap_m
        return self.robot.feasibility(candidate,obj,scene,slot,**kwargs)

    def _assess(self,feasibility):
        provider=None
        if self.grasp_source is not None:
            def refresh():
                self.observe('after_model_inference')
                return self.packet,self.scene
            rows=self.grasp_source.generate(self.packet,self.scene,self.robot,refresh)
            provider=lambda obj,states:[c for c in rows if c.target_id==obj.track_id]
        return assess(self.scene,self.robot.read_states(),feasibility,self.robot.home,
            failure_history=self.failure_history,eligible=self.eligible,candidate_provider=provider,
            accepted_postures=self.accepted_postures)

    def _refresh_candidate(self,candidate,obj):
        if candidate.source=='graspgen':
            from workstation.planning.graspgen_candidates import revalidate_candidate
            return revalidate_candidate(candidate,obj,self.scene,
                {n:f.sequence_id for n,f in self.packet.cameras.items() if f},self.accepted_postures)
        return replace(candidate,target_id=obj.track_id,position_m=tuple(upright_grasp_position(obj.position_m)),
                       uncertainty_m=obj.uncertainty_m)

    def observe(self,label=None):
        packet=self.read_packet()
        self.packet=packet
        self.scene=self.estimator.estimate(packet)
        # Identity must be reconciled BEFORE building this frame's workspace
        # and adding it to measured history. No second model inference is needed.
        measurement=self._read_holding_surface()
        if measurement is not None and hasattr(self.estimator,'associate_supported_target'):
            self.scene=self.estimator.associate_supported_target(self.scene,self.candidate.target_id,measurement)
        self.robot.set_observation(packet,self.scene,self.priors)
        measurement=self._register_holding_surface(measurement)
        if (measurement is not None and 'height_gain_m' in measurement and
                not measurement.get('grasp_constrained',False)):
            # Phase-triggered captures count too: a short lift may finish before
            # the next periodic exposure, but still has fresh independent proof.
            self.max_visual_lift=max(self.max_visual_lift,measurement['height_gain_m'])
        self._hold_observation=(packet,self.scene,self.hold_tracker,measurement) if self.hold_tracker is not None and not self.releasing else None
        self._anchor_normal_stage(label,measurement)
        self.trace_scene(packet,self.scene)
        if label is not None: self.save_scene(label,packet,self.scene)
        # A phase-triggered exposure also satisfies the periodic observer.
        self._monitor_ticks=0
        return self.scene

    def _anchor_normal_stage(self,label,measurement):
        # Fresh rays acquired by NORMAL task motion, with the original 30 s
        # age/4-anchor limits. Never retimestamp old frames or move a camera.
        if (label in ('pregrasp','lifted') and (not self.held or measurement is not None)
                and hasattr(self.robot,'anchor_observation')):
            self.robot.anchor_observation()
            self.observation_anchors.append(dict(phase=label,time_s=self.packet.observation_time_s,
                current_payload_supported=bool(self.held and measurement is not None)))

    def _read_holding_surface(self):
        if not self.held or self.releasing or self.hold_tracker is None:return None
        tcp,rotation=self.robot.fk(self.arm,self.packet.robot_state[self.arm].joint_positions_rad)
        return self.hold_tracker.observe(self.packet,self.scene,tcp,rotation)

    def _register_holding_surface(self,measurement):
        if measurement is not None and hasattr(self.robot,'confirm_payload_observation'):
            if not self.robot.confirm_payload_observation(self.candidate.target_id,self.hold_tracker,measurement,self.packet):
                return None
        return measurement

    def holding_measurement(self):
        """Revalidate payload support after EVERY fresh workspace replacement.

        Cache only this packet/scene/tracker triple, never a previous frame's
        support. Explicit phase observations must follow the same rule as the
        periodic motion monitor, before the next path check uses the workspace.
        """
        if not self.held or self.releasing or self.hold_tracker is None:
            self._hold_observation=None
            return None
        cached=self._hold_observation
        if cached is not None and all(a is b for a,b in zip(cached[:3],(self.packet,self.scene,self.hold_tracker))):
            return cached[3]
        measurement=self._register_holding_surface(self._read_holding_surface())
        self._hold_observation=(self.packet,self.scene,self.hold_tracker,measurement)
        return measurement

    def event(self,phase):
        event=dict(phase=phase,time_s=self.scene.observation_time_s if self.scene else None)
        if self.arm:
            event['measured_fingers_m']=self.robot.read_states()[self.arm].finger_positions_m.tolist()
            if hasattr(self.robot,'targets'):
                event['commanded_fingers_m']=self.robot.targets[self.arm][7:].tolist()
        self.events.append(event)
        print('[visual-pnp] '+phase,flush=True)

    def _monitor_period_steps(self):
        if getattr(self,'grasp_source',None) is None:return 24
        period_s=(getattr(self,'observation_period_s',None) or min(.1,self.priors.max_age_s*2/3)
                  if self.accepted_postures==('UPRIGHT',) else self.priors.max_age_s*.5)
        return max(1,int(period_s/self.robot.dt+1e-9))

    def monitor(self):
        self._monitor_ticks+=1
        period=self._monitor_period_steps()
        if self._monitor_ticks % period: return
        scene=self.observe()
        if not self.held or self.releasing: return
        # Compare against encoder FK at the IMAGE time, not execution's latest q.
        tcp,rotation=self.robot.fk(self.arm,self.packet.robot_state[self.arm].joint_positions_rad)
        if self.hold_tracker is not None:
            measurement=self.holding_measurement()
            if measurement is not None:
                self._missing=0
                if not measurement.get('grasp_constrained',False):
                    self.max_visual_lift=max(self.max_visual_lift,measurement['height_gain_m'])
                self.events[-1]['last_visual_hold']=measurement
                return
        reliable=[o for o in scene.objects if not o.failures and pose_is_observed(o,scene)
                  and o.posture in self.accepted_postures and o.uncertainty_m <= .008]
        # A centre inferred from clipped geometry is not a lift measurement.
        # If the target remains visible at its original support while the robot
        # rises, distinguish physical grip failure from an occluded observation.
        stayed=[o for o in reliable if np.linalg.norm(np.asarray(o.position_m)-self.initial_position) < .012]
        if stayed and tcp[2]-self.initial_position[2] > .04:
            raise RuntimeError('VISUAL_GRIP_FAILED:carton_remains_on_support')
        if self.hold_tracker is not None:
            # A failed current surface check cannot be overridden by a biased
            # complete-looking centre or by closed finger encoders.
            self._missing+=1
            if self._missing >= 3: raise RuntimeError('HELD_CARTON_NOT_OBSERVABLE')
            return
        nearby=[o for o in reliable if np.linalg.norm(np.asarray(o.position_m)-tcp) < .060]
        if not nearby:
            self._missing+=1
            if self._missing >= 3: raise RuntimeError('HELD_CARTON_NOT_OBSERVABLE')
            return
        self._missing=0
        obj=min(nearby,key=lambda o:np.linalg.norm(np.asarray(o.position_m)-tcp))
        gain=obj.position_m[2]-self.initial_position[2]
        self.max_visual_lift=max(self.max_visual_lift,gain)
        # This uses sensed geometry; finger encoders alone never prove holding.
        self.events[-1]['last_visual_hold']=dict(time_s=obj.time_s,track_id=obj.track_id,
            height_gain_m=gain,box_tcp_distance_m=float(np.linalg.norm(np.asarray(obj.position_m)-tcp)),
            visibility=obj.visibility,failures=obj.failures)

    def move(self,position,yaw0,yaw1,speed=None,contact=False,planning_scene=None,orientation=None):
        if self.held and self.hold_tracker is not None and self.holding_measurement() is None:
            raise RuntimeError('CURRENT_HELD_SURFACE_UNSUPPORTED')
        for attempt in range(3):
            try:
                self.robot.execute_cartesian(self.arm,np.asarray(position),yaw0,yaw1,self.scene if planning_scene is None else planning_scene,
                    self.candidate.target_id,speed,self.priors.carton_size_m if self.held else None,contact,
                    check_approach_corridor=self.events[-1]['phase']=='APPROACH',
                    **({'orientation_wxyz':orientation} if orientation is not None else {}))
                break
            except RuntimeError as exc:
                if (str(exc)!='UNOBSERVED_PATH_SPACE' or self.held or attempt==2
                        or not getattr(self.robot,'allow_active_reobserve',False)
                        or self.events[-1]['phase']!='APPROACH' or not hasattr(self.robot,'execute_camera_view')):raise
                point=self.robot.last_plan_check.get('workspace',{}).get('last_unknown_sample_m')
                rejected=self.robot.last_plan_check.copy()
                objects=[o for o in self.scene.objects if o.track_id==self.candidate.target_id and not o.failures]
                if point is None or len(objects)!=1:raise
                from workstation.skills.camera_reobserve import GapReobserve
                self.scene,report=GapReobserve(self.robot,self.observe,lambda:self.packet,self.priors).run(
                    self.arm,point,objects[0],self.scene)
                self.events[-1].setdefault('path_reobservation',[]).append(report)
                report['original_approach_rejection']=rejected
                if report['executed']:self.observer_arms.add(report['observer_arm'])
                if not report['success']:
                    self.robot.last_plan_check=rejected
                    raise
                current=[o for o in self.scene.objects if o.track_id==self.candidate.target_id and
                         not o.failures and o.posture in self.accepted_postures]
                if len(current)!=1 or np.linalg.norm(np.asarray(current[0].position_m)-self.initial_position)>.015:
                    raise RuntimeError('POST_INSPECTION_TARGET_UNCERTAIN')
                self.candidate=self._refresh_candidate(self.candidate,current[0])
                updated=np.asarray(self.candidate.position_m)
                position=updated
                check=self._check_task(self.candidate,current[0],self.scene,self.slot_xy)
                self.events[-1]['post_inspection_task_check']=check
                if check['reasons']:raise RuntimeError('POST_INSPECTION_TASK_NOT_FEASIBLE:'+check['reasons'][0])
                q=check.get('pregrasp_joint_positions_rad')
                if q is not None and np.max(np.abs(np.asarray(q)-self.robot.read_states()[self.arm].joint_positions_rad))>.04:
                    self.robot.execute_joint(self.arm,np.r_[q,[.04,.04]],self.scene,self.candidate.target_id)
                    self.observe('post_inspection_pregrasp')
        self.events[-1]['path_check']=self.robot.last_plan_check.copy()

    def run(self):
        result=dict(success=False,scope=self.run_scope,skill=self.skill_name,
            online_source='RGB-D SceneEstimate + encoder FK',
            box_attachment=False,box_pose_teleport=False,
            acceptance=dict(visual_lift_m=.08,xy_m=.018,z_m=.008,axis_yaw_deg=8,
                            observed_stable_s=.5,release_and_home=True),events=self.events)
        try:
            self.observe('initial')
            if hasattr(self.robot,'anchor_observation'):self.robot.anchor_observation()
            self.robot.hold(.2)
            self.observe('tracked')
            if hasattr(self.robot,'anchor_observation'):self.robot.anchor_observation()
            free=[]
            def refresh_slots():
                nonlocal free
                slot_evidence=[self.robot.observed_workspace.slot(xy) for xy in self.slots]
                result['slot_evidence']=slot_evidence
                free=[tuple(item['xy']) for item in slot_evidence if item['free']]
            refresh_slots()
            def full_task(candidate,obj,scene):
                if self.grasp_source is not None:refresh_slots()
                if not free:return dict(reasons=('NO_FREE_SLOT_OBSERVED',),joint_cost=0.,path_clearance_m=0.)
                destination=min(free,key=lambda xy:np.linalg.norm(np.asarray(candidate.position_m[:2])-xy))
                return self._check_task(candidate,obj,scene,destination)
            assessments=self._assess(full_task)
            result['assessments']=[asdict(a) for a in assessments]
            result['decision']=select_skill(assessments)
            chosen=next((a for a in assessments if a.feasible),None)
            if self.grasp_source is not None:
                result['candidate_attempts']=[result['assessments']]
                # Bounded fresh-observation attempts, never a geometry fallback.
                for attempt in range(2):
                    if chosen is not None:break
                    self.robot.hold(.1);self.observe('model_retry_'+str(attempt+1))
                    assessments=self._assess(full_task)
                    result['assessments']=[asdict(a) for a in assessments]
                    result['candidate_attempts'].append(result['assessments'])
                    result['decision']=select_skill(assessments)
                    chosen=next((a for a in assessments if a.feasible),None)
            decision=result['decision']
            categories=set(decision.get('reason_categories',{}).values())
            if self.grasp_source is None and chosen is None and decision['skill'] == 'REOBSERVE' and categories <= {
                    'PERCEPTION_UNCERTAIN','OCCLUSION','OBSERVED_MOTION','UNOBSERVED_SPACE','PATH_OR_COLLISION'}:
                def refreshed(scene):
                    nonlocal assessments
                    refresh_slots()
                    assessments=self._assess(full_task)
                    return any(a.feasible for a in assessments)
                self.scene,result['reobservation']=Reobserve(self.robot,self.observe).run(refreshed)
                result['assessments']=[asdict(a) for a in assessments]
                result['decision']=select_skill(assessments)
                chosen=next((a for a in assessments if a.feasible),None)
                if chosen is None and getattr(self.robot,'allow_active_reobserve',False) and hasattr(self.robot,'observation_prefix') and any(
                        'UNOBSERVED_PATH_SPACE' in a.reasons for a in assessments):
                    self.scene,result['guided_reobservation']=GuidedReobserve(self.robot,self.observe).run(
                        refreshed,self.scene,lambda:[a.candidate for a in assessments
                                                    if a.reasons == ('UNOBSERVED_PATH_SPACE',)])
                    result['assessments']=[asdict(a) for a in assessments]
                    result['decision']=select_skill(assessments)
                    chosen=next((a for a in assessments if a.feasible),None)
                if chosen is None and getattr(self.robot,'allow_active_reobserve',False):
                    self.scene,result['active_reobservation']=ActiveReobserve(self.robot,self.observe).run(
                        refreshed,self.scene)
                    result['assessments']=[asdict(a) for a in assessments]
                    result['decision']=select_skill(assessments)
                    chosen=next((a for a in assessments if a.feasible),None)
                if chosen is None and not getattr(self.robot,'allow_active_reobserve',False):
                    result['active_reobservation']=dict(executed=False,success=False,
                        reason='FIXED_CAMERA_CONSTRAINT',scope='fresh passive observations only; no dedicated wrist viewpoint motion')
            if chosen is None:
                raise RuntimeError('NO_FEASIBLE_DIRECT_GRASP')
            candidate=chosen.candidate; self.candidate=candidate; self.arm=candidate.arm
            result['decision']['skill']=self.skill_name
            model=candidate.source=='graspgen'
            if self.skill_name=='SIDE_ADJUST' and not model:
                raise RuntimeError('SIDE_REQUIRES_MODEL_POSE')
            if model:
                self.robot.hold(.025);self.observe('before_model_pregrasp')
                current=[o for o in self.scene.objects if o.track_id==candidate.target_id]
                if len(current)!=1:raise RuntimeError('MODEL_TARGET_UNCERTAIN')
                candidate=self._refresh_candidate(candidate,current[0]);self.candidate=candidate
            obj=next(o for o in self.scene.objects if o.track_id == candidate.target_id)
            self.initial_position=np.asarray(obj.position_m)
            result['initial_scene']=self.scene.to_dict()
            # Slot occupancy comes from visual estimates, not the conveyor truth monitor.
            slot_evidence=[self.robot.observed_workspace.slot(xy) for xy in self.slots]
            result['slot_evidence']=slot_evidence
            free=[tuple(item['xy']) for item in slot_evidence if item['free']]
            if not free: raise RuntimeError('NO_FREE_SLOT_OBSERVED')
            slot=min(free,key=lambda xy:np.linalg.norm(np.asarray(candidate.position_m[:2])-xy))
            result['slot_xy']=slot
            self.slot_xy=slot
            prepared_q=None
            if getattr(self.robot,'preposition_observer',False):
                from workstation.skills.camera_reobserve import GapReobserve
                q=np.asarray(chosen.terms['pregrasp_joint_positions_rad'])
                p,r=self.robot.fk(self.arm,q,'panda_link5')
                model=self.robot.collision_links['link5']
                point=np.asarray(p)+np.asarray(r)@((model.lower+model.upper)/2)
                self.scene,preparation=GapReobserve(self.robot,self.observe,lambda:self.packet,self.priors).run(
                    self.arm,point,obj,self.scene,prepared=True)
                result['observer_preparation']=preparation
                if preparation['executed']:self.observer_arms.add(preparation['observer_arm'])
                if not preparation['success']:raise RuntimeError('OBSERVER_PREPARATION_NOT_VERIFIED')
                current=[o for o in self.scene.objects if o.track_id==candidate.target_id and
                    not o.failures and o.posture in self.accepted_postures and pose_is_observed(o,self.scene) and
                    o.valid_depth_ratio>=.9 and o.uncertainty_m<=.008 and o.velocity_m_s is not None and
                    np.linalg.norm(o.velocity_m_s)<=.015 and
                    np.linalg.norm(np.asarray(o.position_m)-self.initial_position)<.015]
                if len(current)!=1:raise RuntimeError('PREPARED_TARGET_UNCERTAIN')
                candidate=self._refresh_candidate(candidate,current[0])
                self.candidate=candidate
                refresh_slots()
                if not free:raise RuntimeError('NO_FREE_SLOT_AFTER_OBSERVER_PREPARATION')
                slot=min(free,key=lambda xy:np.linalg.norm(np.asarray(candidate.position_m[:2])-xy))
                self.slot_xy=slot;result['slot_xy']=slot
                # Keep the previously checked redundant IK branch as a warm
                # start after millimetre-scale visual updates. The measured
                # current robot -> new solution and all task phases are STILL
                # checked against the newly positioned observer and RGB-D.
                check=self._check_task(candidate,current[0],self.scene,slot,pregrasp_hint=q)
                result['post_observer_task_check']=check
                result['prepared_candidate']=asdict(candidate)
                if check['reasons']:raise RuntimeError('PREPARED_TASK_NOT_FEASIBLE:'+check['reasons'][0])
                prepared_q=check['pregrasp_joint_positions_rad']
            pick=np.asarray(candidate.position_m)
            raised=pick-np.asarray(candidate.approach_direction)*candidate.pregrasp_distance_m
            yaw=candidate.yaw_rad
            self.event('PREGRASP')
            q=prepared_q if prepared_q is not None else chosen.terms.get('pregrasp_joint_positions_rad')
            if q is None:
                if model:q=self.robot.solve_pose(self.arm,raised,candidate.orientation_wxyz,self.robot.read_states()[self.arm].joint_positions_rad)
                else:q=self.robot.solve(self.arm,raised,yaw,self.robot.read_states()[self.arm].joint_positions_rad)
            # Open fully for approach; estimated opening is a fit requirement.
            # Closing preload below is computed separately for each model pose.
            self.robot.execute_joint(self.arm,np.r_[q,[.04,.04]],self.scene,candidate.target_id)
            self.events[-1]['path_check']=self.robot.last_plan_check.copy()
            # Re-estimate after approach; do not blindly reuse a moved/ambiguous target.
            scene=self.observe('pregrasp')
            visible=[o for o in scene.objects if not o.failures and o.posture in self.accepted_postures
                     and np.linalg.norm(np.asarray(o.position_m)-self.initial_position) < .015]
            if len(visible) != 1: raise RuntimeError('PREGRASP_TARGET_UNCERTAIN')
            self.candidate=self._refresh_candidate(candidate,visible[0])
            pick=np.asarray(self.candidate.position_m)
            result['executed_candidate']=asdict(self.candidate)
            attitude=self.candidate.orientation_wxyz if model else None
            self.event('APPROACH'); self.move(pick,yaw,yaw,.05,True,orientation=attitude)
            self.event('CLOSE')
            self.robot.grip(self.arm,self.candidate.closing_width_m/2 if model else .018,self.scene,self.candidate.target_id)
            self.held=True
            # Closing encoders do not prove a grip. Require fresh, complete
            # visual geometry before starting the guarded lift experiment.
            closed=self.observe('closed')
            nearby=[o for o in closed.objects if not o.failures and o.posture in self.accepted_postures
                    and pose_is_observed(o,closed) and o.uncertainty_m <= .008
                    and np.linalg.norm(np.asarray(o.position_m)-(self.initial_position if model else pick)) < .015
                    and (not model or o.track_id==self.candidate.target_id)]
            if len(nearby) != 1: raise RuntimeError('CLOSED_TARGET_UNCERTAIN')
            if model:
                # Projecting the entire tilted cuboid can overestimate the
                # actual pad-contact width. Add bounded position preload from
                # image-time finger feedback; the existing 30 N limit stays.
                from workstation.planning.graspgen_candidates import finger_preload_target
                fingers=self.packet.robot_state[self.arm].finger_positions_m
                preload=finger_preload_target(fingers,self.candidate.closing_width_m)
                result['model_preload']=dict(measured_fingers_m=fingers.tolist(),
                    original_command_m=self.candidate.closing_width_m/2,command_m=preload,
                    maximum_position_preload_m=.004,holding_proof=False)
                self.event('PRELOAD');self.robot.grip(self.arm,preload,self.scene,self.candidate.target_id)
                closed=self.observe('preloaded')
                nearby=[o for o in closed.objects if o.track_id==self.candidate.target_id and
                    not o.failures and o.posture in self.accepted_postures and pose_is_observed(o,closed) and
                    o.uncertainty_m<=.008 and np.linalg.norm(np.asarray(o.position_m)-self.initial_position)<.015]
                if len(nearby)!=1:raise RuntimeError('PRELOADED_TARGET_UNCERTAIN')
            tcp,rotation=self.robot.fk(self.arm,self.packet.robot_state[self.arm].joint_positions_rad)
            self.release_tracker=(HeldSurfaceTracker(nearby[0],tcp,rotation,self.priors.carton_size_m)
                                  if nearby[0].posture=='UPRIGHT' else None)
            self.hold_tracker=(CuboidHoldingTracker(nearby[0],tcp,rotation,self.priors.carton_size_m,self.priors,
                              allow_grasp_translation=self.accepted_postures==('UPRIGHT',))
                               if model else self.release_tracker)
            self.robot.held_visual_geometry=dict(arm=self.arm,target_id=self.candidate.target_id,
                allow_grasp_translation=bool(model and self.accepted_postures==('UPRIGHT',)),
                offset=self.hold_tracker.offset.copy(),relative_rotation=self.hold_tracker.relative_rotation.copy(),
                uncertainty_m=self.hold_tracker.uncertainty_m,
                source='pre-lift current observed visual pose and image-time encoder FK')
            result['held_visual_geometry']=dict(offset_tcp_m=self.hold_tracker.offset.tolist(),
                relative_rotation=self.hold_tracker.relative_rotation.tolist(),observation_time_s=closed.observation_time_s,
                uncertainty_m=self.hold_tracker.uncertainty_m,
                source=self.robot.held_visual_geometry['source'],physical_attachment=False)
            measurement=self.hold_tracker.observe(self.packet,closed,tcp,rotation)
            if measurement is None:raise RuntimeError('CLOSED_PAYLOAD_SURFACE_UNSUPPORTED')
            if hasattr(self.robot,'confirm_payload_observation'):
                if not self.robot.confirm_payload_observation(self.candidate.target_id,self.hold_tracker,measurement,self.packet):
                    raise RuntimeError('CLOSED_PAYLOAD_HISTORY_UNSUPPORTED')
            result['closed_surface_support']=measurement
            # Lift from the sensed closed TCP, retaining its actual XY and
            # attitude instead of first correcting a stale nominal grasp pose.
            self.event('LIFT'); self.move(tcp+[0,0,.18],yaw,yaw,.06,True,
                orientation=quaternion_from_matrix(rotation) if model else None)
            self.observe('lifted')
            if self.max_visual_lift < .08: raise RuntimeError('NO_VISUAL_LIFT_PROOF')
            self.event('TRANSFER')
            target=np.array([*slot,self.priors.support_z_m+self.priors.carton_size_m[2]/2+self.robot.place_tcp_clearance_m])
            place_attitude=None
            place_yaw=nearest_aligned_yaw(yaw) if not model else 0.
            if model:
                measurement=self.holding_measurement()
                if measurement is None:raise RuntimeError('CURRENT_HELD_SURFACE_UNSUPPORTED')
                relation=self.robot.held_visual_geometry
                desired_body=nearest_upright_slot_rotation(measurement['rotation'])
                place_rotation=desired_body@np.asarray(relation['relative_rotation']).T
                body_target=np.array([*slot,self.priors.support_z_m+self.priors.carton_size_m[2]/2+self.release_gap_m])
                target=body_target-place_rotation@relation['offset']
                place_attitude=quaternion_from_matrix(place_rotation)
                result['model_placement']=dict(target_tcp_m=target.tolist(),orientation_wxyz=place_attitude.tolist(),
                    desired_body_rotation=desired_body.tolist(),axis_symmetry_rad=math.pi,
                    release_gap_m=self.release_gap_m,
                    visual_offset_tcp_m=relation['offset'].tolist(),relative_rotation=relation['relative_rotation'].tolist(),
                    observation_time_s=self.scene.observation_time_s)
            else:
                result['geometry_placement']=dict(yaw_rad=place_yaw,from_yaw_rad=yaw,axis_symmetry_rad=math.pi)
            self.move(target+[0,0,.18],0. if model else yaw,place_yaw,orientation=place_attitude)
            self.observe('transferred')
            self.event('LOWER'); self.move(target,place_yaw,place_yaw,.05,True,orientation=place_attitude)
            self.observe('lowered')
            release_measurement=self.holding_measurement()
            if release_measurement is None:raise RuntimeError('CURRENT_HELD_SURFACE_UNSUPPORTED')
            tcp,rotation=self.robot.fk(self.arm,self.packet.robot_state[self.arm].joint_positions_rad)
            if self.release_tracker is None:
                self.release_tracker=HeldSurfaceTracker.from_measured_flat_pose(
                    release_measurement,tcp,rotation,self.scene.observation_time_s)
                result['release_semantics']='unresolved until post-release current RGB verification'
            relation=self.robot.held_visual_geometry if model else dict(offset=self.hold_tracker.offset,relative_rotation=self.hold_tracker.relative_rotation)
            released_centre=tcp+rotation@relation['offset']
            released_rotation=rotation@relation['relative_rotation']
            bottom=released_centre[2]-np.abs(released_rotation[2])@(self.hold_tracker.size/2)
            drop_bound=max(0.,float(bottom-self.priors.support_z_m))+self.hold_tracker.uncertainty_m
            if drop_bound>.020:raise RuntimeError('RELEASE_HEIGHT_UNSUPPORTED')
            result['release_geometry']=dict(expected_lowest_corner_m=float(bottom),
                observed_supported_body=True,drop_bound_m=drop_bound,source='current supported visual body and image-time FK; known fixed support')
            self.event('OPEN'); self.releasing=True
            self.robot.grip(self.arm,.04,self.scene,self.candidate.target_id,held_size=self.hold_tracker.size)
            self.held=False;self.releasing=False
            self.observe('released')
            tcp,rotation=self.robot.fk(self.arm,self.packet.robot_state[self.arm].joint_positions_rad)
            fitted=self.release_tracker.collision_estimate(self.packet,self.scene,tcp,rotation,
                max_drop_m=drop_bound,reference_position_m=released_centre)
            if fitted is None: raise RuntimeError('RELEASED_GEOMETRY_UNCERTAIN')
            result['released_collision_estimate']=fitted
            self.scene=self.estimator.associate_supported_target(self.scene,self.candidate.target_id,fitted)
            measured=next(o for o in self.scene.objects if o.track_id == self.candidate.target_id)
            c,s=math.cos(fitted['axis_yaw_rad']),math.sin(fitted['axis_yaw_rad'])
            rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            bounded=replace(measured,position_m=fitted['position_m'],axis_yaw_rad=fitted['axis_yaw_rad'],
                posture='UNKNOWN',visibility='OBSERVED_TOP_AND_SIDE',failures=(),uncertainty_m=.003,
                hypotheses=tuple(PoseHypothesis(label,tuple(quaternion_from_matrix(rz@flip)),
                    'measured flat cuboid; top/bottom semantics unresolved') for label,flip in
                    [('UPRIGHT',np.eye(3)),('INVERTED',np.diag([1.,-1.,-1.]))]),
                quality={**measured.quality,'geometric_length_m':self.priors.carton_size_m[0],
                    'geometric_width_m':self.priors.carton_size_m[1],'observed_height_m':self.priors.carton_size_m[2]})
            collision_scene=replace(self.scene,objects=tuple(bounded if o.track_id == measured.track_id else o
                                                            for o in self.scene.objects))
            # Current measured released geometry replaces this track's clipped
            # proxy for BOTH robot collision and depth-obstacle accounting.
            self.scene=collision_scene
            self.robot.set_observation(self.packet,self.scene,self.priors)
            self.event('RETREAT'); self.move(target+[0,0,.18],place_yaw,place_yaw,.06,True,collision_scene,orientation=place_attitude)
            self.event('HOME')
            self.robot.execute_joint(self.arm,self.robot.home,self.scene,self.candidate.target_id)
            for observer in sorted(self.observer_arms):
                self.observe('before_observer_home_'+observer)
                self.robot.execute_joint(observer,self.robot.home,self.scene,self.candidate.target_id)
                self.observe('observer_home_'+observer)
            self.event('VERIFY')
            proof=[]; stable_start=None
            for index in range(26):
                self.robot.hold(.2); scene=self.observe()
                good=[o for o in scene.objects if not o.failures and o.posture == 'UPRIGHT'
                      and np.linalg.norm(np.asarray(o.position_m[:2])-slot) <= .018]
                ok=False
                if len(good) == 1:
                    o=good[0]; yaw_error=math.degrees(axis_error(o.axis_yaw_rad,0.))
                    z_error=abs(o.position_m[2]-(self.priors.support_z_m+o.size_m[2]/2))
                    speed=float('inf') if o.velocity_m_s is None else np.linalg.norm(o.velocity_m_s)
                    states=self.robot.read_states()
                    clear=all(np.max(np.abs(s.joint_positions_rad-self.robot.home[:7])) < .04
                              and min(s.finger_positions_m) > .037 for s in states.values())
                    ok=yaw_error <= 8 and z_error <= .008 and speed <= .015 and clear
                    proof.append(dict(time_s=scene.observation_time_s,track_id=o.track_id,
                        position_m=o.position_m,axis_yaw_error_deg=yaw_error,z_error_m=z_error,
                        visual_speed_m_s=float(speed),arms_home_and_open=clear))
                if ok:
                    if stable_start is None: stable_start=scene.observation_time_s
                    if scene.observation_time_s-stable_start >= .5:
                        result['success']=True; break
                else: stable_start=None
            result.update(visual_verification=proof,final_scene=self.scene.to_dict(),
                          observed_stable_s=0. if stable_start is None else self.scene.observation_time_s-stable_start)
            self.observe('final')
            if not result['success']: raise RuntimeError('VISUAL_PLACEMENT_NOT_VERIFIED')
            self.event('DONE')
        except Exception as exc:
            result['failure_reason']=str(exc)
            if self.packet is not None and self.scene is not None:
                try:self.save_scene('failure_trigger',self.packet,self.scene)
                except Exception as save_error:result['failure_capture_error']=str(save_error)
            if self.hold_tracker is not None:
                result['holding_geometry_failure']=getattr(self.hold_tracker,'last_failure',None)
            if hasattr(self.robot,'last_plan_check'):
                result['failure_path_check']=self.robot.last_plan_check.copy()
            # Stop at measured robot joints and preserve requested finger force.
            # An uncertain load is neither released nor moved to a guessed pose.
            self.robot.monitor=None
            for name,state in self.robot.read_states().items():
                self.robot.targets[name][:7]=state.joint_positions_rad
            self.robot.hold(.2)
            result['recovery']=dict(action='HOLD_AND_REOBSERVE' if self.held else 'STOP_AND_REOBSERVE',
                                    executed=True,active_camera_motion=False,hold_verified=False,
                                    reason='Measured-joint safe hold, retained finger command and fresh images')
            self.observe('failed')
            self.event('FAILED')
        finally:
            self.robot.monitor=None
            result['max_visual_lift_m']=self.max_visual_lift
            result['held_on_exit']=self.held
            result['normal_stage_observation_anchors']=self.observation_anchors
            if self.grasp_source is not None:
                result['grasp_backend']='graspgen'
                result['model_requests']=self.grasp_source.requests
        return result
