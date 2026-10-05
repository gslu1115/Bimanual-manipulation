"""Fixed inverted carton, physical dual-Panda handover; simulator truth + IK.

No object attachments or live object pose writes. IK checks reachability and
continuity, not full-scene collision avoidance. This is a fixture experiment.
"""
import json
import math
import numpy as np
from pxr import Usd, UsdPhysics
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, load_supported_lula_kinematics_solver_config
from task_logic import rotation, orientation_error, slot_errors, slot_valid, single_pnp_success
from handover_math import multiply, inverse, turn_y, angle_between, slerp
from single_pick_place import blend


class Arm:
    def __init__(self, env, name, p):
        self.env, self.name, self.p = env, name, p
        self.arm = env.arms[name]
        self.solver = LulaKinematicsSolver(**load_supported_lula_kinematics_solver_config('Franka'))
        base = next(b for b in env.c['robots'] if b['name'] == name)
        yaw = math.radians(base['yaw_deg'])/2
        self.solver.set_robot_base_pose(np.array(base['base_xyz']), np.array([math.cos(yaw), 0, 0, math.sin(yaw)]))
        if self.arm.dof_names != self.solver.get_joint_names()+['panda_finger_joint1', 'panda_finger_joint2']:
            raise RuntimeError('Unexpected Panda joint ordering')
        paths = [str(prim.GetPath()) for prim in Usd.PrimRange(env.stage.GetPrimAtPath('/World/Robots/'+name))
                 if prim.GetName() == 'panda_hand' and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(paths) != 1: raise RuntimeError('Expected exactly one physical hand')
        self.hand = RigidPrim(paths)
        self.arm.set_dof_max_efforts([p['grip_max_force']]*2, dof_indices=[7, 8])
        self.arm.set_dof_gains(stiffnesses=[2000., 2000.], dampings=[100., 100.], dof_indices=[7, 8])
        self.target = self.joints()
        self.binding = None

    def joints(self):
        return self.arm.get_dof_positions().numpy()[0].astype(float)

    def pose(self):
        pos, quat = self.hand.get_world_poses()
        q = quat.numpy()[0].astype(float)
        # Lula right_gripper is rotated pi around Z relative to panda_hand.
        # Their +Z/TCP position agree, but raw hand quaternion is NOT the IK frame.
        return pos.numpy()[0]+np.array(rotation(q))@np.array([0, 0, .1]), multiply(q, [0., 0., 0., 1.])

    def solve(self, position, orientation, warm):
        q, ok = self.solver.compute_inverse_kinematics('right_gripper', np.asarray(position), np.asarray(orientation),
                    warm_start=warm, position_tolerance=.0003, orientation_tolerance=.005)
        if not ok: raise RuntimeError(f'IK failed for {self.name} at {np.asarray(position).tolist()}')
        return np.asarray(q)


class DualHandover:
    def __init__(self, env):
        self.env, self.c = env, env.c
        self.p, self.dt = self.c['dual_handover'], self.c['physics']['dt']
        self.arms = {role: Arm(env, self.p[role], self.p) for role in ('giver', 'receiver')}
        self.initial = env.observe_ground_truth()[0]
        self.phase, self.events, self.max_lift = 'OBSERVE', [], 0.
        self.trace, self.owner = None, None
        self.handover_verified = False
        self.video, self.video_frames = None, 0
        self.result = dict(success=False, mode='dual_handover', observation_source='simulator_ground_truth',
                           box_attachment=False, box_pose_teleport=False, controller='Lula IK + physical position drives',
                           direction_semantics='top +Z up; tape axis modulo 180 degrees',
                           initial_box=self.initial.to_dict())
        env.planner_name = 'fixed_inverted_carton_dual_handover'

    def stage(self, name):
        self.phase = name
        self.events.append(dict(phase=name, time_s=self.env.time))
        if self.trace: self.trace.flush()
        print('[handover] '+json.dumps(self.events[-1]), flush=True)

    def tick(self):
        if not self.env.app.is_running(): raise RuntimeError('Application closed during handover')
        for a in self.arms.values(): self.env.command_joints(a.name, a.target)
        self.env.step()
        b = self.env.observe_ground_truth()[0]
        self.max_lift = max(self.max_lift, b.position[2]-self.initial.position[2])
        if b.position[2] < self.c['table']['top_z']-.03:
            raise RuntimeError('Carton fell below the workstation')
        if self.owner is not None:
            a = self.arms[self.owner]
            tcp, q = a.pose()
            rel_p, rel_q = a.binding
            expected = tcp+np.array(rotation(q))@rel_p
            drift = float(np.linalg.norm(np.asarray(b.position)-expected))
            rot_error = math.degrees(angle_between(b.orientation_wxyz, multiply(q, rel_q)))
            if drift > self.p['slip_position_m'] or rot_error > self.p['slip_angle_deg']:
                raise RuntimeError(f'Grip slip: {self.owner}, translation {drift:.4f} m, rotation {rot_error:.2f} deg')
        if self.trace and self.env.ticks % 12 == 0:
            self.trace.write(json.dumps(dict(time_s=self.env.time, phase=self.phase, owner=self.owner, box=b.to_dict(),
                arms={r: dict(joints=a.joints().tolist(), targets=a.target.tolist(), tcp=a.pose()[0].tolist())
                      for r, a in self.arms.items()}))+'\n')
        if self.video and self.env.ticks % 12 == 0:
            from PIL import Image, ImageDraw
            frames = [np.asarray(self.env.sensors[name]['annotators']['rgb'].get_data()) for name in ('overhead', 'oblique')]
            if any(frame.ndim != 3 or frame.shape[2] != 4 for frame in frames):
                raise RuntimeError('Missing RGBA frame during requested recording')
            canvas = Image.fromarray(np.concatenate(frames, axis=1))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, canvas.width, 26), fill=(15, 20, 25, 255))
            draw.text((8, 6), f'{self.phase} | simulation {self.env.time:.2f} s | physical contact, simulator truth', fill='white')
            if not self.video.encode_next_frame_from_buffer(np.asarray(canvas).copy()):
                raise RuntimeError('Video encoder rejected a frame')
            self.video_frames += 1
        return b

    def hold(self, seconds):
        for _ in range(math.ceil(seconds/self.dt)): self.tick()

    def grip(self, role, position):
        a = self.arms[role]
        start = a.joints()[7:]
        steps = math.ceil(1./self.dt)
        for i in range(1, steps+1):
            a.target[7:] = start+blend(i/steps)*(position-start)
            self.tick()
        self.hold(.4)

    def bind_measurement(self, role):
        # Store a measurement for slip detection ONLY; no USD/PhysX constraint.
        a = self.arms[role]
        tcp, q = a.pose()
        b = self.env.observe_ground_truth()[0]
        a.binding = (np.array(rotation(q)).T@(np.asarray(b.position)-tcp), multiply(inverse(q), b.orientation_wxyz))

    def move(self, role, pos, quat, speed=None):
        a = self.arms[role]
        start, ori = a.pose()
        pos, quat = np.asarray(pos), np.asarray(quat)
        duration = max(.7, 1.875*np.linalg.norm(pos-start)/(speed or self.p['move_speed']),
                       1.875*angle_between(ori, quat)/self.p['angular_speed'])
        count = max(2, math.ceil(duration/(4*self.dt)))
        knots = [a.joints()[:7]]
        for i in range(1, count+1):
            s = blend(i/count)
            q = a.solve(start+s*(pos-start), slerp(ori, quat, s), knots[-1])
            if np.max(np.abs(q-knots[-1])) > .15:
                raise RuntimeError(f'IK discontinuity for {role} at knot {i}/{count}, delta {np.max(np.abs(q-knots[-1])):.4f}; segment rejected before motion')
            knots.append(q)
        for q0, q1 in zip(knots, knots[1:]):
            # Slow down high joint-rate portions near singularities rather than
            # applying a large IK increment within the nominal four ticks.
            steps = max(4, math.ceil(np.max(np.abs(q1-q0))/self.p['joint_speed']/self.dt))
            for j in range(1, steps+1):
                a.target[:7] = q0+(q1-q0)*j/steps
                self.tick()
        self.hold(.3)
        actual, aq = a.pose()
        error = float(np.linalg.norm(actual-pos))
        angular = math.degrees(angle_between(aq, quat))
        self.events[-1].update(tcp_error_m=error, tcp_angle_error_deg=angular)
        if error > .012 or angular > 6:
            raise RuntimeError(f'{role} tracking error {error:.4f} m / {angular:.2f} deg')

    def joint_move(self, role, target):
        a = self.arms[role]
        start, target = a.joints(), np.asarray(target)
        steps = math.ceil(max(1., 1.875*np.max(np.abs(target[:7]-start[:7]))/self.p['joint_speed'])/self.dt)
        for i in range(1, steps+1):
            a.target = start+blend(i/steps)*(target-start)
            self.tick()
        self.hold(.4)
        if np.max(np.abs(a.joints()[:7]-target[:7])) > .04: raise RuntimeError('Joint tracking failed')

    def capture(self, name):
        self.env.capture(name)
        self.env.export_snapshot(name+'.usda')

    def receiver_pregrasp(self, contact, orientation, pre):
        # Select the redundant IK branch from the constrained contact pose,
        # then trace backwards to pregrasp. Both jaw-symmetric rolls are valid.
        a = self.arms['receiver']
        home = np.asarray(self.c['robot_home'])[:7]
        candidates, rejected = [], []
        for roll in ([1., 0, 0, 0], [0., 0, 0, 1.]):
            ori = multiply(orientation, roll)
            for swivel in (0., -.7, .7):
                warm = home.copy()
                warm[2] += swivel
                try:
                    q = a.solve(contact, ori, warm)
                    contact_q = q.copy()
                    for i in range(1, 81):
                        next_q = a.solve(contact+(pre-contact)*i/80, ori, q)
                        if np.max(np.abs(next_q-q)) > .15: raise RuntimeError('approach branch discontinuity')
                        q = next_q
                    # Keep the selected jaw roll through the subsequent turn;
                    # otherwise a symmetric grip can cause an unnecessary half turn.
                    final_ori = multiply(turn_y(0), roll)
                    turn_start = contact+np.array([0, 0, .04])
                    turn_end = contact+np.array([.10, 0, .07])
                    turn_q = a.solve(turn_start, ori, contact_q)
                    for i in range(1, 101):
                        s = i/100
                        next_q = a.solve(turn_start+s*(turn_end-turn_start), slerp(ori, final_ori, s), turn_q)
                        if np.max(np.abs(next_q-turn_q)) > .15: raise RuntimeError('upright turn branch discontinuity')
                        turn_q = next_q
                    candidates.append((float(np.linalg.norm(q-home)), q, ori, final_ori))
                except RuntimeError as exc:
                    rejected.append(str(exc))
        self.result['receiver_approach_candidates'] = dict(valid=len(candidates), rejected=rejected)
        if not candidates: raise RuntimeError('No continuous receiver approach found for either symmetric jaw orientation')
        _, q, ori, self.receiver_final_orientation = min(candidates, key=lambda item: item[0])
        self.joint_move('receiver', np.r_[q, [.04, .04]])
        return ori

    def run(self):
        self.env.safe_to_index = False
        self.trace = (self.env.output/'control_trace.jsonl').open('w', encoding='utf-8')
        try:
            if getattr(self.env, 'record_video', False):
                from video_encoding import get_video_encoding_interface
                self.video = get_video_encoding_interface()
                if not self.video.start_encoding(str(self.env.output/'handover.mp4'), 1/(12*self.dt), 0, False):
                    self.video = None
                    raise RuntimeError('Could not initialize requested MP4 recording')
            self.execute()
        except Exception as exc:
            self.result['success'] = False
            self.result['error'] = str(exc)
            self.result['failure_phase'] = self.phase
            self.env.task.last_action_success = False
            # Freeze the timeline, retaining existing grip drive targets.
            try: self.capture('failed_handover')
            except Exception as capture_exc: self.result['capture_error'] = str(capture_exc)
            self.env.timeline.pause()
            print('[handover] FAILED: '+str(exc), flush=True)
        finally:
            if self.video:
                self.video.finalize_encoding()
                self.result['recording'] = dict(file='handover.mp4', frames=self.video_frames, fps=1/(12*self.dt))
            self.trace.close()
            self.result.update(events=self.events, last_phase=self.phase, max_lift_m=self.max_lift,
                               handover_verified=self.handover_verified, simulation_time_s=self.env.time,
                               final_box=self.env.observe_ground_truth()[0].to_dict())
            self.env._write_json('task_report.json', self.result)
            self.env._write_json('run_report.json', self.env.report())
            self.env.export_snapshot('final.usda')
        return self.result

    def execute(self):
        if len(self.env.box_paths) != 1 or orientation_error(self.initial.orientation_wxyz)[0] < 175:
            raise RuntimeError('Handover fixture requires exactly one inverted carton')
        self.env.task.decide([self.initial.name], [])
        for role, a in self.arms.items():
            error = float(np.linalg.norm(a.pose()[0]-a.solver.compute_forward_kinematics('right_gripper', a.joints()[:7])[0]))
            self.result[role+'_model_error_m'] = error
            if error > .005: raise RuntimeError('URDF/USD hand geometry mismatch')
        giver, receiver = self.arms['giver'], self.arms['receiver']
        down = turn_y(0.)
        pick = np.asarray(self.initial.position)+[self.p['giver_offset_x'], 0, .002]
        raised = pick+[0, 0, self.p['lift_height']]
        self.stage('GIVER_PREGRASP')
        q = giver.solve(raised, down, giver.joints()[:7])
        self.joint_move('giver', np.r_[q, [.04, .04]])
        self.stage('GIVER_APPROACH')
        self.move('giver', pick, down, .05)
        self.stage('GIVER_CLOSE')
        self.grip('giver', self.p['grip_position'])
        self.bind_measurement('giver')
        self.owner = 'giver'
        self.stage('GIVER_LIFT')
        self.move('giver', raised, down, .06)
        if self.env.observe_ground_truth()[0].position[2]-self.initial.position[2] < .08:
            raise RuntimeError('Giver failed physical lift')
        self.capture('lifted')
        center = np.asarray(self.p['handover_center'])
        self.stage('GIVER_TRANSFER')
        self.move('giver', center+[self.p['giver_offset_x'], 0, 0], down)
        self.stage('GIVER_TURN')
        sideways = turn_y(-math.pi/2)
        self.move('giver', center+[0, 0, self.p['giver_offset_x']], sideways)
        b = self.env.observe_ground_truth()[0]
        # Receiver grips the opposite half along the long axis; approach from +X.
        receive_pos = np.asarray(b.position)+np.array(rotation(b.orientation_wxyz))@np.array([self.p['receiver_offset_x'], 0, 0])
        receive_q = multiply(b.orientation_wxyz, [0., 0., 1., 0.])
        pre = receive_pos+[self.p['approach_distance'], 0, 0]
        self.stage('RECEIVER_PREGRASP')
        receive_q = self.receiver_pregrasp(receive_pos, receive_q, pre)
        self.stage('RECEIVER_APPROACH')
        self.move('receiver', receive_pos, receive_q, .035)
        self.stage('RECEIVER_CLOSE')
        self.grip('receiver', self.p['grip_position'])
        fingers = receiver.joints()[7:]
        if not all(self.p['grip_position']+.002 < f < .038 for f in fingers):
            raise RuntimeError('Receiver fingers did not establish a carton-width grip')
        self.bind_measurement('receiver')
        self.capture('both_gripping')
        self.stage('GIVER_RELEASE')
        self.owner = 'receiver'
        self.grip('giver', .04)
        self.stage('GIVER_RETREAT')
        gp, gq = giver.pose()
        self.move('giver', gp+[-self.p['approach_distance'], 0, 0], gq, .05)
        self.stage('RECEIVER_PROOF')
        rp, rq = receiver.pose()
        proof_start = self.env.observe_ground_truth()[0].position[2]
        self.move('receiver', rp+[0, 0, .04], rq, .035)
        self.hold(.6)
        if not all(f > .037 for f in giver.joints()[7:]): raise RuntimeError('Giver is not released')
        proof_box = self.env.observe_ground_truth()[0]
        proof_gain = proof_box.position[2]-proof_start
        giver_clearance = float(np.linalg.norm(np.asarray(proof_box.position)-giver.pose()[0]))
        if proof_gain < .025 or giver_clearance < .10:
            raise RuntimeError('Receiver proof failed: no independent physical lift clear of the giver')
        self.handover_verified = True
        self.result['handover_proof'] = dict(giver_fingers=giver.joints()[7:].tolist(),
                                           receiver_fingers=receiver.joints()[7:].tolist(), proof_lift_m=proof_gain,
                                           giver_tcp_distance_m=giver_clearance)
        self.capture('handover_verified')
        self.stage('GIVER_HOME')
        self.joint_move('giver', self.c['robot_home'])
        self.stage('RECEIVER_UPRIGHT')
        rp, _ = receiver.pose()
        receiver_down = self.receiver_final_orientation
        self.move('receiver', rp+[.10, 0, .03], receiver_down)
        self.capture('upright')
        xy = self.c['conveyor']['slots_xy'][self.p['slot_index']]
        box_target = np.array(xy+[self.c['conveyor']['top_z']+self.c['boxes']['size'][2]/2+.004])
        tcp_target = box_target-np.array(rotation(receiver_down))@receiver.binding[0]
        self.stage('RECEIVER_TRANSFER')
        self.move('receiver', tcp_target+[0, 0, self.p['lift_height']], receiver_down)
        self.stage('RECEIVER_LOWER')
        self.move('receiver', tcp_target, receiver_down, .045)
        self.stage('RECEIVER_RELEASE')
        self.owner = None
        self.grip('receiver', .04)
        self.capture('released')
        self.stage('RECEIVER_RETREAT')
        self.move('receiver', tcp_target+[0, 0, self.p['lift_height']], receiver_down, .06)
        self.stage('RECEIVER_HOME')
        self.joint_move('receiver', self.c['robot_home'])
        self.env.task.action_finished()
        self.stage('VERIFY')
        stable, success = 0., False
        for _ in range(math.ceil(self.p['verify_timeout_s']/self.dt)):
            b = self.tick()
            errors = slot_errors(b, xy, self.c)
            stable = stable+self.dt if slot_valid(errors, self.c) else 0.
            home = all(np.max(np.abs(a.joints()[:7]-np.asarray(self.c['robot_home'])[:7])) < .04 for a in self.arms.values())
            success = self.handover_verified and all(single_pnp_success(errors, self.c, self.max_lift, stable,
                                      a.joints()[7:], home) for a in self.arms.values())
            if success: break
        self.result.update(success=bool(success), slot_errors=errors, stable_duration_s=stable, both_arms_home=bool(home))
        self.env.task.verified(success)
        if not success: raise RuntimeError('Final placement/release/home verification timed out')
        self.env.release_conveyor_interlock()
        self.env.task.decide([], [])
        self.stage('DONE')
        self.capture('final')
