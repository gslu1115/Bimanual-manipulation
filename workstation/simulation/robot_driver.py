"""Robot-only joint drive port and Lula model; no carton or contact truth access."""
from __future__ import annotations
import math
import numpy as np
from workstation.observations.observation_packet import PandaJointMap, PANDA_ARM_JOINT_NAMES, PANDA_FINGER_JOINT_NAMES
from workstation.planning.collision import AABB, check_spheres, station_obstacles, scene_obstacles, sphere_box_clearance
from workstation.planning.selection import UPRIGHT_TCP_HEIGHT_M, UPRIGHT_PLACE_TCP_CLEARANCE_M


def smooth(t): return t*t*t*(10.+t*(-15.+6.*t))


def down(yaw): return np.array([0.,math.cos(yaw/2),math.sin(yaw/2),0.])


class RobotMotion:
    def __init__(self, config, read_states, command, step):
        # Callables expose encoders, position drive commands and time stepping only.
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, load_supported_lula_kinematics_solver_config
        import yaml
        self.read_states,self.command,self.step = read_states,command,step
        self.dt=float(config['physics']['dt'])
        self.home=np.asarray(config['robot_home'],float)
        self.speed=float(config['single_pick_place']['move_speed'])
        self.joint_speed=float(config['single_pick_place']['joint_speed'])
        self.static=station_obstacles(config)
        self.place_support_z=float(config['conveyor']['top_z'])
        # Keep a controlled body release gap plus the nominal TCP/body offset.
        # All paths retain optical-depth and held-body uncertainty checks.
        # The released carton settles by gravity/contact, without a pose write.
        self.place_tcp_clearance_m=UPRIGHT_PLACE_TCP_CLEARANCE_M
        self.solvers={}
        description=load_supported_lula_kinematics_solver_config('Franka')
        from pathlib import Path
        geometry=yaml.safe_load(Path(description['robot_description_path']).read_text())
        self.sphere_description=[]
        for group in geometry['collision_spheres']:
            for link,items in group.items():
                index=int(link[-1]) if link.startswith('panda_link') else 8
                for sphere in items:
                    self.sphere_description.append((link,index,np.asarray(sphere['center']),float(sphere['radius'])))
        for robot in config['robots']:
            solver=LulaKinematicsSolver(**description)
            yaw=math.radians(robot['yaw_deg'])/2
            solver.set_robot_base_pose(np.asarray(robot['base_xyz']),np.array([math.cos(yaw),0,0,math.sin(yaw)]))
            if tuple(solver.get_joint_names()) != PANDA_ARM_JOINT_NAMES:
                raise RuntimeError('Lula joint names require an explicit mapping')
            self.solvers[robot['name']]=solver
        self.limits=next(iter(self.solvers.values())).get_cspace_position_limits()
        self.targets={name:state.qpos.copy() for name,state in self.read_states().items()}
        self.plan_cache={}
        self.last_plan_check={}
        self.monitor=None
        self.observed_workspace=None
        self.require_observed_workspace=True
        self.pregrasp_attempts=9
        from workstation.planning.occlusion_memory import OcclusionMemory
        self.occlusion_memory=OcclusionMemory()
        # User's saved camera installation is fixed. Dedicated observer moves
        # remain opt-in diagnostic code, never an automatic fallback.
        self.allow_active_reobserve=False
        self.preposition_observer=bool(config.get('diagnostic_preposition_observer',False))
        import os
        from workstation.planning.robot_geometry import load_panda_collision_models
        self.collision_links=load_panda_collision_models(os.environ.get('ISAAC_PATH','D:/Issaccc'))

    def set_observation(self,packet,scene,priors):
        from workstation.planning.observed_workspace import ObservedWorkspace
        image_spheres=[];image_links=[]
        for arm,state in packet.robot_state.items():
            image_spheres.extend(self.spheres(arm,state.joint_positions_rad,
                                              float(np.mean(state.finger_positions_m))))
            for i in range(8):
                p,r=self.fk(arm,state.joint_positions_rad,'panda_link'+str(i))
                image_links.append((self.collision_links['link'+str(i)],np.asarray(p),np.asarray(r)))
            p,r=self.fk(arm,state.joint_positions_rad,'panda_hand');p,r=np.asarray(p),np.asarray(r)
            image_links.append((self.collision_links['hand'],p,r))
            for index,sign in enumerate((1.,-1.)):
                finger_p=p+r@np.array([0.,sign*state.finger_positions_m[index],.0584])
                finger_r=r if sign>0 else r@np.diag([-1.,-1.,1.])
                image_links.append((self.collision_links['finger'],finger_p,finger_r))
        workspace=ObservedWorkspace(packet,priors,scene,self.static,image_spheres,robot_links=image_links)
        if hasattr(self,'occlusion_memory'):
            self.occlusion_memory.add(workspace)
            workspace.occlusion_memory=self.occlusion_memory
        self.observed_workspace=workspace

    def anchor_observation(self):
        if self.observed_workspace is not None:self.occlusion_memory.anchor(self.observed_workspace)

    def confirm_payload_observation(self,target_id,tracker,measurement,packet):
        if self.observed_workspace.packet is not packet:return False
        visual=self.held_visual_geometry
        if visual is None or visual['target_id']!=target_id:return False
        tcp,rotation=self.fk(visual['arm'],packet.robot_state[visual['arm']].joint_positions_rad)
        measured=bool(measurement.get('geometric_pose_measured'))
        if measured:
            # Workspace validates the proof and revokes previous same-frame
            # confirmation on failure, including incomplete measured geometry.
            centre=np.asarray(measurement.get('position_m'),float)
            body_rotation=np.asarray(measurement.get('rotation'),float)
            uncertainty=max(tracker.uncertainty_m,measurement.get('uncertainty_m',float('nan')))
        else:
            centre=np.asarray(tcp)+np.asarray(rotation)@tracker.offset
            body_rotation=np.asarray(rotation)@tracker.relative_rotation
            uncertainty=getattr(tracker,'uncertainty_m',.003)
        confirmed=self.observed_workspace.confirm_payload(target_id,centre,body_rotation,tracker.size,measurement,uncertainty)
        if confirmed and measured:
            # Planning predicts future sweep from the CURRENT measured body/TCP
            # relation. The observer keeps its original latch to detect drift.
            visual.update(offset=np.asarray(rotation).T@(centre-np.asarray(tcp)),
                relative_rotation=np.asarray(rotation).T@body_rotation,uncertainty_m=uncertainty,
                measured_pose_time_s=packet.observation_time_s)
        return confirmed

    def fk(self,arm,q=None,frame='right_gripper'):
        q=self.read_states()[arm].joint_positions_rad if q is None else q
        return self.solvers[arm].compute_forward_kinematics(frame,np.asarray(q))

    def collision_poses(self,arm,q,fingers=(.04,.04),finger_sweep=False):
        for i in range(8):
            p,r=self.fk(arm,q,'panda_link'+str(i))
            yield 'link'+str(i),self.collision_links['link'+str(i)],np.asarray(p),np.asarray(r)
        p,r=self.fk(arm,q,'panda_hand');p,r=np.asarray(p),np.asarray(r)
        yield 'hand',self.collision_links['hand'],p,r
        for index,sign in enumerate((1.,-1.)):
            model=self.collision_links['finger']
            opening=fingers[index]
            if finger_sweep:
                model=model.extruded([0.,-.04,0.]);opening=.04
            yield 'finger'+str(index),model,p+r@np.array([0.,sign*opening,.0584]),r if sign>0 else r@np.diag([-1.,-1.,1.])

    def unknown_body_diagnostic(self,arm,q,point):
        if point is None or not hasattr(self,'collision_links'):return {}
        margins={};bounds={}
        for name,model,p,r in self.collision_poses(arm,q):
            local=(np.asarray(point)-p)@r
            margins[name]=float(np.max(model.planes[:,:3]@local+model.planes[:,3]))
            bounds[name]=float(np.max(np.maximum(model.lower-local,local-model.upper)))
        return dict(candidate_joint_positions_rad=np.asarray(q).tolist(),
                    unknown_point_max_plane_distance_m=margins,
                    unknown_point_local_bbox_overshoot_m=bounds,
                    unknown_point_inside_any_convex_body_3mm=any(d<=.003+1e-8 and bounds[name]<=.003+1e-8 for name,d in margins.items()))

    def solve(self,arm,position,yaw,warm):
        return self.solve_pose(arm,position,down(yaw),warm)

    def solve_pose(self,arm,position,orientation,warm):
        q,ok=self.solvers[arm].compute_inverse_kinematics('right_gripper',np.asarray(position),np.asarray(orientation),
                    warm_start=np.asarray(warm),position_tolerance=.0003,orientation_tolerance=.005)
        if not ok: raise RuntimeError('ROBOT_UNREACHABLE')
        q=np.asarray(q)
        if np.any(q < self.limits[0]+.005) or np.any(q > self.limits[1]-.005):
            raise RuntimeError('JOINT_LIMIT')
        return q

    def solve_pregrasp(self,arm,position,yaw,start,scene,target_id,pregrasp_hint=None):
        """Try bounded redundant IK seeds; goal bodies must be observed and clear."""
        attempts=getattr(self,'pregrasp_attempts',1)
        seeds=[np.asarray(start).copy()]
        if pregrasp_hint is not None:
            hint=np.asarray(pregrasp_hint,float)
            if hint.shape!=(7,) or not np.all(np.isfinite(hint)):
                raise RuntimeError('INVALID_PREGRASP_HINT')
            if np.any(hint<self.limits[0]+.005) or np.any(hint>self.limits[1]-.005):
                raise RuntimeError('JOINT_LIMIT')
            seeds.insert(0,hint.copy())
        if attempts==1:return self.solve(arm,position,yaw,seeds[0])
        for joint,offset in ((2,.6),(2,-.6),(4,.5),(4,-.5),(1,.35),(1,-.35),(0,.4),(0,-.4)):
            warm=np.asarray(start).copy();warm[joint]+=offset
            seeds.append(np.clip(warm,self.limits[0]+.005,self.limits[1]-.005))
        records=[];solutions=[];last='ROBOT_UNREACHABLE'
        for index,warm in enumerate(seeds[:attempts]):
            try:
                q=self.solve(arm,position,yaw,warm)
                if any(np.max(np.abs(q-old))<.015 for old in solutions):continue
                solutions.append(q)
                self.validate_path(arm,[q,q],scene,target_id)
                self.last_plan_check['ik_seed_index']=index
                return q
            except RuntimeError as exc:
                last=str(exc);records.append(dict(seed_index=index,reason=last,details=self.last_plan_check.copy()))
        if records:
            last=next((r['reason'] for r in records if r['reason']=='UNOBSERVED_PATH_SPACE'),
                      next((r['reason'] for r in records if r['reason']!='ROBOT_UNREACHABLE'),last))
        self.last_plan_check=dict(rejection=last,pregrasp_ik_attempts=records,
                                  unreachable_proven=False)
        raise RuntimeError(last)

    def observation_prefix(self,arm,goal,scene,target_id,max_delta=.10):
        """Choose a fully observed EMPTY-arm prefix; never move through unknown."""
        start=self.read_states()[arm].joint_positions_rad
        delta=np.asarray(goal)-start;distance=float(np.max(np.abs(delta)))
        if distance<.0125:raise RuntimeError('VIEWPOINT_ALREADY_REACHED')
        records=[]
        for limit in (max_delta,max_delta/2,max_delta/4,max_delta/8):
            q=start+delta*min(1.,limit/distance)
            try:
                self.validate_path(arm,self.joint_knots(start,q),scene,target_id)
                return q,dict(max_joint_delta_rad=float(np.max(np.abs(q-start))),rejected_prefixes=records)
            except RuntimeError as exc:
                records.append(dict(max_joint_delta_rad=limit,reason=str(exc)))
        self.last_plan_check=dict(rejection='NO_OBSERVED_VIEWPOINT_PREFIX',prefixes=records)
        raise RuntimeError('NO_OBSERVED_VIEWPOINT_PREFIX')

    def execute_camera_view(self,arm,frame,camera_position,aim,scene,target_id):
        from workstation.observations.camera_geometry import pose_matrix,look_at_quaternion,quaternion_from_matrix
        from workstation.planning.camera_viewpoint import rigid_transform,camera_goal_to_tcp
        self.motion_started=False
        image_q=frame.robot_state_at_frame[arm].joint_positions_rad
        p,r=self.fk(arm,image_q,'panda_hand');hand=rigid_transform(p,r)
        p,r=self.fk(arm,image_q);tcp=rigid_transform(p,r)
        goal=pose_matrix(camera_position,look_at_quaternion(camera_position,aim,[0,0,1]))@np.diag([1.,-1.,-1.,1.])
        target,T_hand_camera=camera_goal_to_tcp(hand,tcp,frame.T_workcell_from_camera_cv,goal)
        q=self.solve_pose(arm,target[:3,3],quaternion_from_matrix(target[:3,:3]),
                          self.read_states()[arm].joint_positions_rad)
        self.execute_joint(arm,np.r_[q,[.04,.04]],scene,target_id)
        p,r=self.fk(arm,frame='panda_hand');measured=rigid_transform(p,r)@T_hand_camera
        error=float(np.linalg.norm(measured[:3,3]-goal[:3,3]))
        angle=float(np.arccos(np.clip((np.trace(measured[:3,:3].T@goal[:3,:3])-1)/2,-1,1)))
        if error>.015 or angle>np.radians(3):raise RuntimeError('INSPECTION_CAMERA_TRACKING_ERROR')
        return dict(camera_position_goal_m=list(camera_position),camera_position_fk_m=measured[:3,3].tolist(),
                    camera_position_error_m=error,camera_angle_error_rad=angle,
                    calibration_source='image-time robot FK and existing calibrated camera frame')

    def execute_observation_lift(self,arm,height,scene):
        from workstation.observations.camera_geometry import quaternion_from_matrix
        self.motion_started=False
        state=self.read_states()[arm];start,rotation=self.fk(arm,state.joint_positions_rad)
        orientation=quaternion_from_matrix(rotation)
        duration=max(1.,1.875*height/.04);count=math.ceil(duration/(4*self.dt))
        knots=[state.joint_positions_rad.copy()]
        for i in range(1,count+1):
            position=start+np.array([0.,0.,height*smooth(i/count)])
            q=self.solve_pose(arm,position,orientation,knots[-1])
            if np.max(np.abs(q-knots[-1]))>.15:raise RuntimeError('IK_DISCONTINUITY')
            knots.append(q)
        self.validate_path(arm,knots,scene,None)
        self.motion_started=True
        for q0,q1 in zip(knots,knots[1:]):
            for k in range(1,5):
                self.targets[arm][:7]=q0+(q1-q0)*k/4;self.tick()
        self.hold(.3)
        actual,_=self.fk(arm)
        if np.linalg.norm(actual-start-[0,0,height])>.012:raise RuntimeError('TCP_TRACKING_ERROR')

    def spheres(self,arm,q,fingers=.04):
        poses={}
        spheres=[]
        for link,index,centre,radius in self.sphere_description:
            if link not in poses: poses[link]=self.fk(arm,q,link)
            p,r=poses[link]
            spheres.append((index,np.asarray(p)+np.asarray(r)@centre,radius))
        p,r=self.fk(arm,q,'panda_hand')
        for sign in (-1,1):
            for z in (.060,.080,.095):
                spheres.append((9,np.asarray(p)+np.asarray(r)@np.array([0,sign*(fingers+.004),z]),.009))
        return spheres

    def validate_path(self,arm,knots,scene,target_id,held_size=None,held_yaw=0.,support_contact=False,target_contact=False,
                      target_boxes_override=None,finger_positions=None,check_local_unknown=True,check_approach_corridor=True):
        states=self.read_states()
        other=next(name for name in states if name != arm)
        other_spheres=self.spheres(other,states[other].joint_positions_rad)
        obstacles=list(self.static)+scene_obstacles(scene,target_id)
        # Selected contact object retains measured position uncertainty. The
        # extra neighbour keep-out padding would erase the narrow Panda palm/
        # box gap even when only fingers are intended to touch. Body checks
        # still require 2 mm clearance and are never exempted.
        target_boxes=[b for b in scene_obstacles(scene,None,extra_margin=0.) if b.name == target_id]
        if target_boxes_override is not None: target_boxes=list(target_boxes_override)
        workspace=getattr(self,'observed_workspace',None)
        if workspace is None and getattr(self,'require_observed_workspace',False):
            raise RuntimeError('MISSING_WORKSPACE_OBSERVATION')
        minimum=float('inf'); samples=0
        # Include actual segment starts and interpolated command trajectory samples.
        for start,end in zip(knots,knots[1:]):
            count=max(1,math.ceil(float(np.max(np.abs(end-start)))/.025))
            for alpha in np.linspace(0,1,count+1):
                q=start+alpha*(end-start); samples+=1
                if np.any(q < self.limits[0]) or np.any(q > self.limits[1]):
                    raise RuntimeError('JOINT_LIMIT')
                fingers=states[arm].finger_positions_m if finger_positions is None else np.asarray(finger_positions)
                own_spheres=self.spheres(arm,q,float(np.max(fingers)))
                reason,clearance=check_spheres(own_spheres,other_spheres,obstacles,arm+'_mount')
                minimum=min(minimum,clearance)
                if reason:
                    self.last_plan_check=dict(rejection=reason,arm=arm,other_arm=other,
                        clearance_m=float(clearance),candidate_joint_positions_rad=q.tolist(),
                        other_arm_joint_positions_rad=states[other].joint_positions_rad.tolist(),
                        source='robot encoder FK, official collision spheres and fixed/visual obstacles')
                    raise RuntimeError(reason)
                if workspace is not None:
                    # Preserve observed obstacles for EVERY sphere and
                    # unknown-space evidence for TCP corridor/held box.
                    # Never-observed whole-arm space is NOT proven safe.
                    queries=[]
                    for link,centre,radius in own_spheres:
                        reason=workspace.check_observed_sphere_obstacles(centre,radius)
                        if reason:queries.append((link,reason));break
                    if not queries and check_local_unknown and check_approach_corridor and target_contact and held_size is None:
                        tcp,_=self.fk(arm,q)
                        reason=workspace.check_sphere(np.asarray(tcp),.003)
                        if reason:queries.append(('TCP_APPROACH_CORRIDOR',reason))
                    for link,reason in queries:
                        if reason:
                            self.last_plan_check=dict(rejection=reason,link_index=link,
                                workspace=workspace.report())
                            self.last_plan_check.update(self.unknown_body_diagnostic(arm,q,workspace.last_unknown_sample))
                            raise RuntimeError(reason)
                if held_size is None:
                    for box in target_boxes:
                        for link,centre,radius in own_spheres:
                            if link == 9 and target_contact: continue
                            if sphere_box_clearance(centre,radius,box) < .002:
                                self.last_plan_check=dict(rejection='TARGET_BODY_COLLISION',link_index=link,
                                    sphere_centre_m=np.asarray(centre).tolist(),sphere_radius_m=radius,
                                    target_centre_m=box.centre,target_size_m=box.size)
                                raise RuntimeError('TARGET_BODY_COLLISION')
                if held_size is not None:
                    tcp,tcp_rotation=self.fk(arm,q)
                    # Swept orientation at EACH FK sample, not destination yaw.
                    # Nominal upright template has a shared TCP/box offset;
                    # visual monitoring must reject slip/unknown holding.
                    visual=getattr(self,'held_visual_geometry',None)
                    if visual is not None and visual['arm']==arm and visual['target_id']==target_id:
                        body_rotation=np.asarray(tcp_rotation)@visual['relative_rotation']
                        centre_box=np.asarray(tcp)+np.asarray(tcp_rotation)@visual['offset']
                        padding=float(visual.get('uncertainty_m',.003))
                    else:
                        # Nominal preflight only. Runtime stores the SAME
                        # measured relation as the visual holding observer.
                        body_rotation=np.asarray(tcp_rotation)@np.diag([1.,-1.,-1.])
                        centre_box=np.asarray(tcp)+np.asarray(tcp_rotation)@np.array([0.,0.,UPRIGHT_TCP_HEIGHT_M])
                        padding=.003
                    if not np.isfinite(padding) or not 0<=padding<=.008:
                        raise RuntimeError('HELD_POSE_UNCERTAINTY_UNBOUNDED')
                    padding=max(.003,padding)
                    local_half=np.asarray(held_size)/2+padding
                    half=np.abs(body_rotation)@local_half
                    if workspace is not None:
                        reason=workspace.check_oriented_box(centre_box,2*local_half,body_rotation,
                                                            check_unknown=check_local_unknown)
                        if reason:
                            self.last_plan_check=dict(rejection=reason,held_centre_m=centre_box.tolist(),
                                held_rotation=body_rotation.tolist(),held_local_size_m=(2*local_half).tolist(),
                                held_space_query='oriented cuboid; enclosing AABB corners excluded',
                                workspace=workspace.report())
                            raise RuntimeError(reason)
                    # Target contact is allowed only at the calibrated support plane.
                    for box in obstacles:
                        gap=np.abs(centre_box-box.centre)-half-np.asarray(box.size)/2
                        if np.all(gap < 0):
                            depth=float(-gap[2])
                            if support_contact and box.name in ('table','belt') and depth <= .007:
                                continue
                            raise RuntimeError('HELD_CARTON_PATH_BLOCKED:'+box.name)
                    for _,centre,radius in other_spheres:
                        local=body_rotation.T@(np.asarray(centre)-centre_box)
                        if np.linalg.norm(np.maximum(np.abs(local)-local_half,0)) < radius+.002:
                            raise RuntimeError('HELD_CARTON_OTHER_ARM_COLLISION')
                    for link,centre,radius in own_spheres:
                        if link == 9: continue  # Intended finger contact only.
                        # Sphere distance to the ORIENTED box. Its enclosing
                        # world AABB creates false palm collisions at yaw != 0.
                        local=body_rotation.T@(np.asarray(centre)-centre_box)
                        clearance=float(np.linalg.norm(np.maximum(np.abs(local)-local_half,0))-radius)
                        if clearance < .002:
                            self.last_plan_check=dict(rejection='HELD_CARTON_OWN_BODY_COLLISION',
                                link_index=link,clearance_m=clearance,
                                sphere_radius_m=float(radius),sphere_local_m=local.tolist(),
                                held_local_half_m=local_half.tolist(),held_padding_m=padding,
                                required_clearance_m=.002,
                                nominal_clearance_m=float(np.linalg.norm(np.maximum(
                                    np.abs(local)-np.asarray(held_size)/2,0))-radius))
                            raise RuntimeError('HELD_CARTON_OWN_BODY_COLLISION')
        self.last_plan_check=dict(samples=samples,clearance_m=minimum,
            model='official Lula spheres + finger proxies + fixed AABBs + observed neighbour AABBs',
            observed_space_body_model='Lula sphere observed-obstacle checks;3mm TCP approach corridor and held-box unknown-space checks',
            whole_arm_unknown_space_verified=False,
            finger_shell_visibility_required=False,
            approach_corridor_unknown_space_checked=bool(check_local_unknown and check_approach_corridor and target_contact and held_size is None),
            approach_corridor_applicable=bool(check_approach_corridor),
            empty_motion_unknown_space_verified=False,
            local_unknown_space_check_deferred=not check_local_unknown,
            max_joint_sample_rad=.025,continuous_collision_proof=False)
        return minimum

    def joint_knots(self,start,target):
        count=max(2,math.ceil(float(np.max(np.abs(target-start)))/.04))
        return [start+smooth(i/count)*(target-start) for i in range(count+1)]

    def checked_joint_plan(self,arm,start,target,scene,target_id,target_boxes_override=None):
        """Empty-arm joint path; every searched edge uses the full existing checks."""
        path=self.joint_knots(np.asarray(start),np.asarray(target))
        def validate(knots):
            return self.validate_path(arm,knots,scene,target_id,
                target_boxes_override=target_boxes_override)
        try:
            clearance=validate(path)
            self.last_plan_check['joint_planner']='quintic straight joint path'
            return path,clearance
        except RuntimeError as exc:
            original=str(exc);detail=self.last_plan_check.copy()
            if not original.startswith(('UNOBSERVED_PATH_SPACE','OBSERVED_PATH_OBSTACLE',
                    'STATIC_PATH_BLOCKED','OTHER_ARM_COLLISION','SELF_COLLISION','TARGET_BODY_COLLISION')):
                raise
        from workstation.planning.joint_rrt import bounded_joint_rrt
        rejected={};minimum=float('inf')
        def edge_valid(q0,q1):
            nonlocal minimum
            try:
                minimum=min(minimum,validate([q0,q1]));return True
            except RuntimeError as exc:
                rejected[str(exc)]=rejected.get(str(exc),0)+1;return False
        path,report=bounded_joint_rrt(start,target,self.limits,edge_valid,
                                      iterations=160,seconds=8.,seed=0)
        report.update(edge_rejections=rejected,straight_path_rejection=original,
                      straight_path_details=detail)
        self.last_joint_search=report
        if path is None:
            self.last_plan_check=dict(rejection=original,joint_search=report)
            raise RuntimeError(original)
        # Full returned-path recheck also accounts for the stationary other arm.
        clearance=validate(path)
        self.last_plan_check.update(joint_planner='bounded joint RRT-Connect',joint_search=report)
        return path,clearance

    def cartesian_knots(self,arm,start,target,yaw0,yaw1,speed=None,warm_start=None,
                        orientation0=None,orientation1=None):
        start=np.asarray(start); target=np.asarray(target)
        duration=max(.7,1.875*np.linalg.norm(target-start)/(speed or self.speed),abs(yaw1-yaw0)/.35)
        if orientation0 is not None or orientation1 is not None:
            from workstation.baselines.handover_math import slerp,angle_between
            orientations=[]
            for value in (orientation0,orientation1):
                q=np.asarray(value,float)
                if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-9:
                    raise ValueError('Cartesian orientations require finite wxyz quaternions')
                orientations.append(q/np.linalg.norm(q))
            duration=max(duration,1.875*angle_between(*orientations)/.35)
        count=max(2,math.ceil(duration/(4*self.dt)))
        # Runtime FK and trajectory start must use the same measured state.
        # Forecast phases pass their planned predecessor explicitly; drive
        # setpoints are not encoder observations and are not IK warm states.
        warm=np.asarray(self.read_states()[arm].joint_positions_rad if warm_start is None else warm_start).copy()
        knots=[warm]
        for i in range(1,count+1):
            a=smooth(i/count)
            if orientation0 is None:
                q=self.solve(arm,start+a*(target-start),yaw0+a*(yaw1-yaw0),knots[-1])
            else:
                q=self.solve_pose(arm,start+a*(target-start),slerp(*orientations,a),knots[-1])
            if np.max(np.abs(q-knots[-1])) > .15: raise RuntimeError('IK_DISCONTINUITY')
            knots.append(q)
        return knots

    def feasibility(self,candidate,obj,scene,slot_xy=None,pregrasp_hint=None):
        start=self.read_states()[candidate.arm].joint_positions_rad
        raised=np.asarray(candidate.position_m)+[0,0,.18]
        phase='PREGRASP'; self.last_plan_check={}
        try:
            q=self.solve_pregrasp(candidate.arm,raised,candidate.yaw_rad,start,scene,candidate.target_id,pregrasp_hint)
            knots,minimum=self.checked_joint_plan(candidate.arm,start,q,scene,candidate.target_id)
            joint_plans={'PREGRASP':self.last_plan_check.copy()}
            # Check straight descent as well, before any movement.
            phase='APPROACH'
            previous=self.targets[candidate.arm].copy()
            self.targets[candidate.arm][:7]=q
            try:
                descend=self.cartesian_knots(candidate.arm,raised,candidate.position_m,candidate.yaw_rad,candidate.yaw_rad,.05,q)
            finally: self.targets[candidate.arm]=previous
            minimum=min(minimum,self.validate_path(candidate.arm,descend,scene,candidate.target_id,target_contact=True))
            phases=['PREGRASP','APPROACH']
            if slot_xy is not None:
                place=np.array([*slot_xy,self.place_support_z+obj.size_m[2]/2+self.place_tcp_clearance_m])
                previous=self.targets[candidate.arm].copy()
                current=descend[-1]; position=np.asarray(candidate.position_m); yaw=candidate.yaw_rad
                try:
                    for phase,destination,new_yaw,held,contact in (
                        ('LIFT',raised,yaw,True,True),
                        ('TRANSFER',place+[0,0,.18],0.,True,False),
                        ('LOWER',place,0.,True,True),
                        ('RETREAT',place+[0,0,.18],0.,False,True)):
                        self.targets[candidate.arm][:7]=current
                        path=self.cartesian_knots(candidate.arm,position,destination,yaw,new_yaw,.05,current)
                        released=None
                        if phase == 'RETREAT':
                            # Forecast the visually estimated carton at its
                            # planned support after release, not back at pickup.
                            # Runtime retreat still requires a NEW observed fit.
                            uncertainty=float(getattr(obj,'uncertainty_m',.003))
                            released=[AABB(candidate.target_id,
                                (*slot_xy,self.place_support_z+obj.size_m[2]/2),
                                tuple(np.asarray(obj.size_m)+2*uncertainty))]
                        minimum=min(minimum,self.validate_path(candidate.arm,path,scene,candidate.target_id,
                            obj.size_m if held else None,new_yaw,contact,not held,target_boxes_override=released,
                            finger_positions=np.full(2,min(.04,obj.size_m[1]/2+obj.uncertainty_m)) if held else np.full(2,.04),
                            check_local_unknown=False,check_approach_corridor=phase!='RETREAT'))
                        current=path[-1];position=destination;yaw=new_yaw;phases.append(phase)
                    phase='HOME'
                    _,home_clearance=self.checked_joint_plan(candidate.arm,current,self.home[:7],
                        scene,candidate.target_id,target_boxes_override=released)
                    minimum=min(minimum,home_clearance)
                    joint_plans['HOME']=self.last_plan_check.copy()
                    phases.append('HOME')
                finally: self.targets[candidate.arm]=previous
            return dict(reasons=(),joint_cost=float(np.linalg.norm(q-start)),path_clearance_m=minimum,
                        checked_phases=phases,whole_task_template_checked=slot_xy is not None,
                        whole_task_check_scope='sampled geometry/collision and currently observed obstacles; future local visibility deferred',
                        pending_observation_phases=['LIFT','TRANSFER','LOWER','RETREAT'] if slot_xy is not None else [],
                        joint_plans=joint_plans,pregrasp_joint_positions_rad=q.tolist(),
                        pregrasp_hint_used=pregrasp_hint is not None)
        except RuntimeError as exc:
            return dict(reasons=(str(exc),),joint_cost=0.,path_clearance_m=0.,
                        rejection_phase=phase,rejection_details=self.last_plan_check.copy())

    def tick(self):
        for name,q in self.targets.items(): self.command(name,q)
        self.step()
        if self.monitor is not None: self.monitor()

    def hold(self,seconds):
        for _ in range(math.ceil(seconds/self.dt)): self.tick()

    def execute_joint(self,arm,target,scene,target_id):
        self.motion_started=False
        start=self.read_states()[arm].qpos
        target=np.asarray(target)
        path,_=self.checked_joint_plan(arm,start[:7],target[:7],scene,target_id)
        planned=self.last_plan_check.copy()
        self.motion_started=True
        if planned.get('joint_planner') == 'bounded joint RRT-Connect':
            lengths=[np.max(np.abs(q1-q0)) for q0,q1 in zip(path,path[1:])]
            total=max(sum(lengths),1e-12);travelled=0.
            for q0,q1,length in zip(path,path[1:],lengths):
                duration=max(.25,1.875*length/self.joint_speed)
                count=math.ceil(duration/self.dt)
                for i in range(1,count+1):
                    a=smooth(i/count)
                    self.targets[arm][:7]=q0+a*(q1-q0)
                    self.targets[arm][7:]=start[7:]+((travelled+a*length)/total)*(target[7:]-start[7:])
                    self.tick()
                travelled+=length
        else:
            duration=max(1.,1.875*np.max(np.abs(target[:7]-start[:7]))/self.joint_speed)
            count=math.ceil(duration/self.dt)
            for i in range(1,count+1):
                self.targets[arm]=start+smooth(i/count)*(target-start); self.tick()
        self.hold(.4)
        self.last_plan_check=planned
        if np.max(np.abs(self.read_states()[arm].joint_positions_rad-target[:7])) > .04:
            raise RuntimeError('JOINT_TRACKING_ERROR')

    def _require_rotating_payload(self,arm,target_id,held_size):
        held_size=np.asarray(held_size,float)
        if held_size.shape!=(3,) or not np.isfinite(held_size).all() or np.any(held_size<=0):
            raise ValueError('Rotating payload requires three positive metric dimensions')
        ws=getattr(self,'observed_workspace',None);visual=getattr(self,'held_visual_geometry',None)
        payload=None if ws is None else ws.payload_observation
        if (payload is None or visual is None or payload['target_id']!=target_id or
                visual['arm']!=arm or visual['target_id']!=target_id or
                not payload.get('geometric_pose_measured',False) or self.monitor is None or
                abs(payload['time_s']-ws.packet.observation_time_s)>1e-6 or
                not 0<=ws.packet.assembled_time_s-payload['time_s']<=ws.priors.max_age_s or
                not np.allclose(payload['size'],held_size,atol=1e-8,rtol=0)):
            raise RuntimeError('CURRENT_ROTATING_PAYLOAD_GEOMETRY_UNSUPPORTED')
        # The cached packet's assembly time does not advance with physics.
        # Execution's encoder timestamp provides the current common clock.
        age=self.read_states()[arm].sample_time_s-payload['time_s']
        if not 0<=age<=ws.priors.max_age_s:
            raise RuntimeError('CURRENT_ROTATING_PAYLOAD_GEOMETRY_STALE')

    def execute_cartesian(self,arm,target,yaw0,yaw1,scene,target_id,speed=None,held_size=None,support_contact=False,check_approach_corridor=True,
                          orientation_wxyz=None):
        """Execute an existing checked path, optionally with a full TCP attitude.

        A full-pose request starts from encoder FK and uses the existing SLERP/
        Lula path. Carrying requires CURRENT independently measured geometry;
        the legacy upright support observer cannot certify a rotating payload.
        """
        self.motion_started=False
        target=np.asarray(target,float)
        if target.shape!=(3,) or not np.isfinite(target).all():raise ValueError('Cartesian target requires finite metric position')
        full_pose=orientation_wxyz is not None
        if full_pose:
            orientation_wxyz=np.asarray(orientation_wxyz,float)
            norm=np.linalg.norm(orientation_wxyz)
            if (orientation_wxyz.shape!=(4,) or not np.isfinite(orientation_wxyz).all() or
                    not np.isfinite(norm) or norm<1e-9):
                raise ValueError('Cartesian orientation requires finite wxyz quaternion')
            orientation_wxyz=orientation_wxyz/norm
            measured=self.read_states()[arm].joint_positions_rad.copy()
            start,rotation=self.fk(arm,measured)
            if held_size is not None:
                self._require_rotating_payload(arm,target_id,held_size)
        else:
            start,rotation=self.fk(arm)
        # Resynchronise the seven arm joints, but retain the requested gripper
        # drive preload. Replacing it by contact-limited measured finger opening
        # removes closing force precisely when lifting begins.
        self.targets[arm][:7]=measured if full_pose else self.read_states()[arm].joint_positions_rad
        preserve_attitude=(not full_pose and held_size is not None and abs(yaw1-yaw0)<1e-9 and
                           np.linalg.norm(np.asarray(target)[:2]-np.asarray(start)[:2])<1e-6)
        if full_pose:
            from workstation.observations.camera_geometry import quaternion_from_matrix
            knots=self.cartesian_knots(arm,start,target,yaw0,yaw1,speed,warm_start=measured,
                orientation0=quaternion_from_matrix(rotation),orientation1=orientation_wxyz)
        elif preserve_attitude:
            from workstation.observations.camera_geometry import quaternion_from_matrix
            orientation=quaternion_from_matrix(rotation)
            # Lift/lower the carried body without an initial nominal-down
            # roll/pitch correction at its support. Retain measured attitude;
            # all generated knots still pass the unchanged full path checks.
            knots=self.cartesian_knots(arm,start,target,yaw0,yaw1,speed,
                                      orientation0=orientation,orientation1=orientation)
        else:knots=self.cartesian_knots(arm,start,target,yaw0,yaw1,speed)
        self.validate_path(arm,knots,scene,target_id,held_size,yaw1,support_contact,held_size is None and support_contact,
                           check_approach_corridor=check_approach_corridor)
        self.last_plan_check['carrying_measured_attitude_preserved']=preserve_attitude
        self.last_plan_check['full_tcp_orientation_requested']=full_pose
        runtime_checks=0
        for q0,q1 in zip(knots,knots[1:]):
            if full_pose:
                if held_size is not None:self._require_rotating_payload(arm,target_id,held_size)
                # Use the newest scene installed by the image-time monitor.
                # Recheck every next edge: dynamic obstacles/payload relation
                # may have changed since the complete preflight.
                ws=getattr(self,'observed_workspace',None)
                current_scene=scene if ws is None else ws.scene
                self.validate_path(arm,[q0,q1],current_scene,target_id,held_size,yaw1,support_contact,
                    held_size is None and support_contact,check_approach_corridor=check_approach_corridor)
                runtime_checks+=1
            for k in range(1,5):
                self.motion_started=True
                self.targets[arm][:7]=q0+(q1-q0)*(k/4); self.tick()
        self.hold(.3)
        if full_pose and held_size is not None:
            self._require_rotating_payload(arm,target_id,held_size)
        p,rotation=self.fk(arm)
        if np.linalg.norm(p-target) > .012: raise RuntimeError('TCP_TRACKING_ERROR')
        if full_pose:
            from workstation.baselines.handover_math import angle_between
            from workstation.observations.camera_geometry import quaternion_from_matrix
            error=float(angle_between(quaternion_from_matrix(rotation),orientation_wxyz))
            self.last_plan_check.update(full_tcp_orientation_requested=True,
                requested_orientation_wxyz=orientation_wxyz.tolist(),endpoint_orientation_error_rad=error,
                runtime_pose_edges_checked=runtime_checks)
            if error>.05:raise RuntimeError('TCP_ORIENTATION_TRACKING_ERROR')

    def grip(self,arm,position,scene=None,target_id=None,held_size=None):
        if not np.isfinite(position) or not 0<=position<=.04:raise ValueError('Invalid Panda finger command')
        if held_size is not None:
            visual=getattr(self,'held_visual_geometry',None);ws=getattr(self,'observed_workspace',None)
            payload=None if ws is None else ws.payload_observation
            if (visual is None or visual['arm']!=arm or visual['target_id']!=target_id or
                    payload is None or payload['target_id']!=target_id or
                    abs(payload['time_s']-ws.packet.observation_time_s)>1e-6 or
                    not 0<=ws.packet.assembled_time_s-payload['time_s']<=ws.priors.max_age_s):
                raise RuntimeError('CURRENT_RELEASE_SURFACE_UNSUPPORTED')
        start=self.read_states()[arm].finger_positions_m
        if scene is not None:
            q=self.read_states()[arm].joint_positions_rad
            count=max(1,math.ceil(float(np.max(np.abs(position-start)))/.001))
            for alpha in np.linspace(0,1,count+1):
                self.validate_path(arm,[q,q],scene,target_id,held_size=held_size,
                                   support_contact=held_size is not None,target_contact=True,
                                   finger_positions=start+alpha*(position-start))
        for i in range(1,121):
            self.targets[arm][7:]=start+smooth(i/120)*(position-start); self.tick()
        self.hold(.3)


def from_environment(env):
    # Native<->canonical name mapping lives only in this simulator drive adapter.
    names=PANDA_ARM_JOINT_NAMES+PANDA_FINGER_JOINT_NAMES
    maps={name:tuple(arm.dof_names.index(joint) for joint in names) for name,arm in env.arms.items()}
    for name,arm in env.arms.items():
        PandaJointMap(tuple(arm.dof_names))
        fingers=list(maps[name][7:])
        arm.set_dof_max_efforts([30.,30.],dof_indices=fingers)
        arm.set_dof_gains(stiffnesses=[2000.,2000.],dampings=[100.,100.],dof_indices=fingers)
    def command(name,q):
        native=np.empty(9); native[list(maps[name])]=q
        env.command_joints(name,native)
    return RobotMotion(env.c,env.observe_robot_states,command,
                       lambda:env.step(monitor_conveyor=False,render=False))
