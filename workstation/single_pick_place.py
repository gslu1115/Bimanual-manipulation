"""Physical single-carton baseline, explicitly using simulator pose observations.

One clear, upright carton; one arm active. Lula IK does not provide collision
planning. The raised waypoints are specific to this workstation fixture.
Only joint-drive targets are commanded; carton transforms are never modified.
"""
from __future__ import annotations

import json
import math
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, load_supported_lula_kinematics_solver_config

from task_logic import rotation, slot_errors, slot_valid, single_pnp_success, orientation_error


def down_orientation(yaw):
    # Rz(yaw) Rx(pi), scalar-first; finger closing direction is world +/-Y at yaw=0.
    return np.array([0., math.cos(yaw/2), math.sin(yaw/2), 0.])


def blend(t):
    return t*t*t*(10. + t*(-15. + 6.*t))


class SinglePickPlace:
    def __init__(self, env):
        self.env, self.c = env, env.c
        self.p = self.c['single_pick_place']
        self.arm = env.arms[self.p['arm']]
        self.dt = self.c['physics']['dt']
        self.solver = LulaKinematicsSolver(**load_supported_lula_kinematics_solver_config('Franka'))
        base = next(r for r in self.c['robots'] if r['name'] == self.p['arm'])
        a = math.radians(base['yaw_deg'])/2
        self.solver.set_robot_base_pose(np.array(base['base_xyz']), np.array([math.cos(a), 0., 0., math.sin(a)]))
        expected = self.solver.get_joint_names() + ['panda_finger_joint1', 'panda_finger_joint2']
        if self.arm.dof_names != expected:
            raise RuntimeError(f'Unexpected joint order: {self.arm.dof_names}; expected {expected}')
        paths = [str(p.GetPath()) for p in Usd.PrimRange(env.stage.GetPrimAtPath('/World/Robots/'+self.p['arm']))
                 if p.GetName() == 'panda_hand' and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(paths) != 1: raise RuntimeError(f'Expected one Panda hand, found {paths}')
        self.hand = RigidPrim(paths)
        # Official asset drive limits are retained except for explicit gripper force/gains.
        self.arm.set_dof_max_efforts([self.p['grip_max_force']]*2, dof_indices=[7, 8])
        self.arm.set_dof_gains(stiffnesses=[2000., 2000.], dampings=[100., 100.], dof_indices=[7, 8])
        self.q_target = self.joints()
        self.phase, self.events = 'OBSERVE', []
        self.max_lift = 0.
        self.initial = env.observe_ground_truth()[0]
        self.last_yaw = 0.
        self.trace = None
        self.result = {'success': False, 'mode': 'single_pick_place', 'observation_source': 'simulator_ground_truth',
                       'controller': 'Lula IK + physical position drives', 'arm': self.p['arm'],
                       'slot_index': self.p['slot_index'], 'object': self.initial.name,
                       'box_attachment': False, 'box_pose_teleport': False}
        env.planner_name = 'single_carton_ground_truth_lula_ik'

    def joints(self):
        return self.arm.get_dof_positions().numpy()[0].astype(float)

    def tcp(self):
        p, q = self.hand.get_world_poses()
        p, q = p.numpy()[0], q.numpy()[0]
        return p + np.array(rotation(q)) @ np.array([0., 0., .1])

    def fk(self):
        return self.solver.compute_forward_kinematics('right_gripper', self.joints()[:7])[0]

    def ik(self, position, yaw, warm):
        q, ok = self.solver.compute_inverse_kinematics('right_gripper', np.asarray(position), down_orientation(yaw),
                    warm_start=warm, position_tolerance=.0003, orientation_tolerance=.005)
        if not ok: raise RuntimeError(f'IK failed during {self.phase}: {position}')
        return np.asarray(q)

    def tick(self):
        if not self.env.app.is_running(): raise RuntimeError('Application closed during task')
        self.env.command_joints(self.p['arm'], self.q_target)
        self.env.step()
        b = self.env.observe_ground_truth()[0]
        self.max_lift = max(self.max_lift, b.position[2]-self.initial.position[2])
        if self.trace is not None and self.env.ticks % 12 == 0:
            self.trace.write(json.dumps({'time_s': self.env.time, 'phase': self.phase, 'joints': self.joints().tolist(),
                               'joint_targets': self.q_target.tolist(), 'tcp_xyz': self.tcp().tolist(),
                               'box': b.to_dict()})+'\n')
        return b

    def hold(self, duration):
        for _ in range(math.ceil(duration/self.dt)): self.tick()

    def phase_start(self, phase):
        self.phase = phase
        if self.trace is not None: self.trace.flush()
        event = {'phase': phase, 'time_s': self.env.time}
        self.events.append(event)
        print('[pnp] '+json.dumps(event), flush=True)

    def joint_move(self, target):
        start = self.joints()
        duration = max(1., 1.875*np.max(np.abs(np.asarray(target)[:7]-start[:7]))/self.p['joint_speed'])
        steps = math.ceil(duration/self.dt)
        for i in range(1, steps+1):
            self.q_target = start + blend(i/steps)*(np.asarray(target)-start)
            self.tick()
        self.hold(.4)
        if np.max(np.abs(self.joints()[:7]-np.asarray(target)[:7])) > .04:
            raise RuntimeError(f'Joint tracking failed at {self.phase}')

    def cartesian_move(self, target, yaw, speed=None):
        start = self.fk()
        target = np.asarray(target)
        yaw_start = self.last_yaw
        duration = max(.7, 1.875*np.linalg.norm(target-start)/(speed or self.p['move_speed']),
                       abs(yaw-yaw_start)/.35)
        # IK knots at <= 30 Hz, physical drives at 120 Hz. All knots are solved
        # before motion; abort unreachable/discontinuous paths without moving.
        count = max(2, math.ceil(duration/(4*self.dt)))
        knots = [self.q_target[:7].copy()]
        for i in range(1, count+1):
            s = blend(i/count)
            q = self.ik(start+s*(target-start), yaw_start+s*(yaw-yaw_start), knots[-1])
            if np.max(np.abs(q-knots[-1])) > .15:
                raise RuntimeError(f'IK discontinuity at {self.phase}; path rejected')
            knots.append(q)
        for q0, q1 in zip(knots, knots[1:]):
            for k in range(1, 5):
                self.q_target[:7] = q0 + (q1-q0)*(k/4)
                b = self.tick()
                if self.phase in ('TRANSFER', 'LOWER') and np.linalg.norm(np.asarray(b.position)-self.tcp()) > .055:
                    raise RuntimeError(f'Carton slipped during {self.phase}')
        self.last_yaw = yaw
        self.hold(.3)
        error = float(np.linalg.norm(self.tcp()-target))
        self.events[-1]['tcp_error_m'] = error
        if error > .012: raise RuntimeError(f'TCP tracking error {error:.4f} m in {self.phase}')

    def gripper(self, finger_position, duration=1.):
        start = self.joints()[7:]
        steps = math.ceil(duration/self.dt)
        for i in range(1, steps+1):
            self.q_target[7:] = start + blend(i/steps)*(finger_position-start)
            self.tick()
        self.hold(.3)

    def run(self):
        self.trace = (self.env.output/'control_trace.jsonl').open('w', encoding='utf-8')
        try:
            self._execute()
        except Exception as exc:
            self.result['error'] = str(exc)
            self.env.task.last_action_success = False
            self.env.task.phase = 'OBSERVE'
            self.env.capture('failed_pnp')
            print('[pnp] FAILED: '+str(exc), flush=True)
        finally:
            self.trace.close()
            self.result.update({'events': self.events, 'last_phase': self.phase, 'max_lift_m': self.max_lift,
                                'simulation_time_s': self.env.time, 'final_box': self.env.observe_ground_truth()[0].to_dict(),
                                'final_finger_positions': self.joints()[7:].tolist()})
            self.env._write_json('task_report.json', self.result)
            self.env._write_json('run_report.json', self.env.report())
            self.env.export_snapshot('final.usda')
        return self.result

    def _execute(self):
        if len(self.env.box_paths) != 1: raise RuntimeError('Single-box controller requires exactly one carton')
        tilt, _ = orientation_error(self.initial.orientation_wxyz)
        if tilt > 5.: raise RuntimeError('Single-box baseline requires an upright carton')
        # Check the URDF against actual PhysX hand pose before relying on its IK.
        model_error = float(np.linalg.norm(self.tcp()-self.fk()))
        self.result['initial_model_tcp_error_m'] = model_error
        if model_error > .005: raise RuntimeError(f'URDF/USD TCP mismatch: {model_error} m')
        targets = self.env.stage.GetPrimAtPath('/World/Targets')
        if not targets or any(p.IsA(UsdGeom.Gprim) for p in Usd.PrimRange(targets)):
            raise RuntimeError('Virtual targets must have no renderable geometry')
        self.result['targets_have_render_geometry'] = False
        self.env.task.decide([self.initial.name], [self.initial.name])
        self.env.safe_to_index = False
        r = rotation(self.initial.orientation_wxyz)
        yaw = math.atan2(r[1][0], r[0][0])
        pick = np.array(self.initial.position)+[0., 0., .002]
        lift = self.p['lift_height']
        self.phase_start('PREGRASP')
        q = self.ik(pick+[0, 0, lift], yaw, self.joints()[:7])
        self.joint_move(np.r_[q, [.04, .04]])
        self.last_yaw = yaw
        self.phase_start('APPROACH')
        self.cartesian_move(pick, yaw, speed=.06)
        self.phase_start('CLOSE')
        self.gripper(self.p['grip_position'])
        self.phase_start('LIFT')
        self.cartesian_move(pick+[0, 0, lift], yaw, speed=.07)
        self.hold(.5)
        b = self.env.observe_ground_truth()[0]
        self.result['lift_check'] = {'height_gain_m': b.position[2]-self.initial.position[2],
                                    'box_tcp_distance_m': float(np.linalg.norm(np.asarray(b.position)-self.tcp())),
                                    'finger_positions': self.joints()[7:].tolist()}
        if self.result['lift_check']['height_gain_m'] < .08 or self.result['lift_check']['box_tcp_distance_m'] > .04:
            raise RuntimeError('Physical lift failed: carton did not follow the closed gripper')
        self.env.capture('lifted')
        slot = self.c['conveyor']['slots_xy'][self.p['slot_index']]
        place = np.array(slot+[self.c['conveyor']['top_z']+self.c['boxes']['size'][2]/2+.004])
        self.phase_start('TRANSFER')
        self.cartesian_move(place+[0, 0, lift], 0.)
        self.phase_start('LOWER')
        self.cartesian_move(place, 0., speed=.05)
        self.phase_start('OPEN')
        self.gripper(.04)
        self.env.capture('released')
        self.phase_start('RETREAT')
        self.cartesian_move(place+[0, 0, lift], 0., speed=.07)
        self.phase_start('HOME')
        self.joint_move(np.array(self.c['robot_home']))
        self.env.task.action_finished()
        self.phase_start('VERIFY')
        stable = 0.
        for _ in range(math.ceil(self.p['verify_timeout_s']/self.dt)):
            b = self.tick()
            errors = slot_errors(b, slot, self.c)
            stable = stable+self.dt if slot_valid(errors, self.c) else 0.
            both_home = all(np.max(np.abs(a.get_dof_positions().numpy()[0][:7]-np.array(self.c['robot_home'])[:7])) < .04
                            for a in self.env.arms.values())
            success = single_pnp_success(errors, self.c, self.max_lift, stable, self.joints()[7:], both_home)
            if success: break
        self.result.update({'success': bool(success), 'slot_errors': errors, 'stable_duration_s': stable,
                            'both_arms_home': bool(both_home)})
        self.env.task.verified(success)
        if success:
            self.env.release_conveyor_interlock()
            self.env.task.decide([], [])
            self.phase_start('DONE')
        else:
            self.result['error'] = 'Placement did not meet released/stable/pose/home verification before timeout'
        self.env.capture('final')
        print('[pnp] '+json.dumps(self.result), flush=True)
