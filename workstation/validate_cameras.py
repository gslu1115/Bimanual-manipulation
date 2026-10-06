"""Isaac Sim camera smoke test, plus optional unchanged fixed-box controller regression."""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

from task_logic import validate


def check(env, full_pick_place=False):
    import numpy as np
    from isaacsim.core.experimental.prims import RigidPrim
    from camera_config import FORMAL_CAMERAS
    from camera_geometry import pose_matrix
    from single_pick_place import SinglePickPlace, blend

    report = dict(success=False, full_physical_pick_place=False, frames={}, motion_checks={})
    hands = {robot: RigidPrim([path]) for robot, path in env.camera_system.hand_paths.items()}

    def frame(label):
        stats = env.capture('check_'+label)
        if set(FORMAL_CAMERAS)-stats.keys(): raise RuntimeError('All three formal cameras must be enabled for validation')
        result = {}
        for name in FORMAL_CAMERAS:
            data, cfg = stats[name], env.camera_system.cameras[name]['config']
            if not all(data[key] is not None for key in ('rgb_shape', 'depth_shape', 'instance_shape')):
                raise RuntimeError(f'Missing RGB/depth/instance output: {name}')
            meta = data['extrinsics']
            if cfg.get('type') == 'wrist':
                pos, quat = hands[cfg['robot']].get_world_poses()
                expected = pose_matrix(pos.numpy()[0], quat.numpy()[0]) @ pose_matrix(
                    cfg['mount_translation'], cfg['mount_orientation_wxyz'])
                actual = np.asarray(meta['T_world_from_camera_usd'])
                error = float(np.max(np.abs(expected-actual)))
                data['hand_mount_matrix_error'] = error
                if error > .005: raise RuntimeError(f'Wrist renderer/physical-hand transform mismatch: {name}: {error}')
            result[name] = data
        report['frames'][label] = result
        env._write_json('camera_validation.json', report)
        print('[camera-frame] '+label, flush=True)
        return {n: np.asarray(result[n]['extrinsics']['T_world_from_camera_usd']) for n in FORMAL_CAMERAS}

    def move(robot, target):
        start = env.arms[robot].get_dof_positions().numpy()[0]
        steps = round(2./env.c['physics']['dt'])
        for i in range(1, steps+1):
            env.command_joints(robot, start+blend(i/steps)*(target-start))
            env.step()
        for _ in range(round(.4/env.c['physics']['dt'])): env.step()

    home = np.asarray(env.c['robot_home'])
    before = frame('HOME_INITIAL')
    for robot, name, other in [('panda_left', 'left_wrist_camera', 'right_wrist_camera'),
                               ('panda_right', 'right_wrist_camera', 'left_wrist_camera')]:
        target = home.copy()
        target[0] += .25 if robot == 'panda_left' else -.25
        move(robot, target)
        changed = frame(robot+'_MOVED')
        motion = float(np.max(np.abs(changed[name]-before[name])))
        unrelated = float(np.max(np.abs(changed[other]-before[other])))
        scene_drift = float(np.max(np.abs(changed['scene_camera']-before['scene_camera'])))
        report['motion_checks'][robot] = dict(wrist_transform_change=motion,
            other_wrist_transform_change=unrelated, scene_transform_change=scene_drift,
            wrist_translation_change_m=float(np.linalg.norm(changed[name][:3, 3]-before[name][:3, 3])),
            other_wrist_translation_change_m=float(np.linalg.norm(changed[other][:3, 3]-before[other][:3, 3])))
        if motion < .02 or unrelated > .005 or scene_drift > 1e-6:
            raise RuntimeError(f'Camera motion independence failed: {robot}')
        move(robot, home)
        before = frame(robot+'_RETURNED_HOME')
    started = time.perf_counter()
    for _ in range(120): env.step()
    wall = time.perf_counter()-started
    report['step_benchmark'] = dict(steps=120, wall_s=wall, physics_steps_per_wall_s=120/wall)
    # Exercise low-frequency observations/periodic metadata without image files.
    # These are camera settings only; PhysX dt/solver and controller are unchanged.
    settings = env.camera_system.settings
    original = settings.copy()
    try:
        settings.update(save_images=False, recording_enabled=True,
                        render_interval_steps=12, capture_interval_steps=24)
        start_renders = env.camera_system.metrics['render_calls']
        start_saves = env.camera_system.metrics['saved_frames']
        started = time.perf_counter()
        for _ in range(48): env.step()
        wall = time.perf_counter()-started
        renders = env.camera_system.metrics['render_calls']-start_renders
        saves = env.camera_system.metrics['saved_frames']-start_saves
        frames = env.capture('check_NO_IMAGE_SAVE')
        arrays_available = all(all(frames[n][k] is not None for k in
            ('rgb_shape', 'depth_shape', 'instance_shape')) for n in FORMAL_CAMERAS)
        image_files = [p.name for p in env.output.glob('check_NO_IMAGE_SAVE_*')
                       if p.suffix in ('.png', '.npy')]
        report['low_frequency_no_image_save'] = dict(success=arrays_available and not image_files and renders == 4 and saves == 2,
            physics_steps=48, render_calls=renders, periodic_captures=saves,
            render_interval_steps=12, capture_interval_steps=24, image_files=image_files,
            wall_s=wall, physics_steps_per_wall_s=48/wall)
        if not report['low_frequency_no_image_save']['success']:
            raise RuntimeError('Low-frequency/no-image-save camera mode failed')
    finally:
        settings.clear()
        settings.update(original)
    if full_pick_place:
        class PhaseCapturingPickPlace(SinglePickPlace):
            def phase_start(self, name):
                # Capture the completed previous phase with physics unchanged.
                # All state transitions, grasp/IK/drive parameters are inherited.
                if self.phase in ('PREGRASP', 'APPROACH', 'CLOSE', 'LIFT', 'LOWER', 'HOME'):
                    frame(self.phase)
                super().phase_start(name)
        result = PhaseCapturingPickPlace(env).run()
        report['single_pick_place'] = result
        report['full_physical_pick_place'] = bool(result['success'])
        if not result['success']: raise RuntimeError('Fixed single-box physical regression failed')
    report.update(success=True, rig=env.camera_system.describe(), performance=env.camera_system.performance())
    env._write_json('camera_validation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--keep-open', action='store_true')
    parser.add_argument('--single-pnp', action='store_true', help='Also run the original fixed-box controller to completion')
    parser.add_argument('--robot-usd')
    parser.add_argument('--no-save-images', action='store_true')
    parser.add_argument('--camera-recording', action='store_true')
    args = parser.parse_args()
    c = json.loads(args.config.read_text(encoding='utf-8'))
    if args.seed is not None: c['seed'] = args.seed
    if args.single_pnp: c['boxes']['count'] = 1
    c['run_mode'] = 'single_pick_place' if args.single_pnp else 'camera_smoke_test'
    if args.no_save_images: c.setdefault('camera_system', {})['save_images'] = False
    if args.camera_recording: c.setdefault('camera_system', {})['recording_enabled'] = True
    validate(c)
    if args.headless and args.keep_open: parser.error('--keep-open requires a GUI')
    from isaacsim import SimulationApp
    root = Path(os.environ.get('ISAAC_PATH', 'D:/isaacsim'))
    app = SimulationApp({'headless': args.headless, 'renderer': 'RaytracedLighting', 'anti_aliasing': 2,
        'multi_gpu': False, 'sync_loads': True, 'extra_args': ['--ext-folder', str(root/'extscache'),
        '--ext-folder', str(root/'extsDeprecated')]}, experience=str(Path(__file__).with_name('sorting.kit')))
    env, failure = None, None
    try:
        from environment import SortingEnvironment
        env = SortingEnvironment(app, c, args.output.resolve(), args.robot_usd, single_pnp=args.single_pnp)
        env.start()
        settled = env.settle()
        if not all(settled[k] for k in ('settled', 'inside_upstream_tray', 'above_table')):
            raise RuntimeError('Fixture did not settle')
        result = check(env, args.single_pnp)
        env._write_json('run_report.json', env.report())
        env._write_json('run_status.json', dict(status='passed', mode=c['run_mode'], camera_validation=True,
                                             full_physical_pick_place=result['full_physical_pick_place']))
        print('[camera-check] '+str(env.output), flush=True)
        while args.keep_open and app.is_running(): app.update()
    except BaseException:
        failure = traceback.format_exc()
        print(failure, flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'failure.txt').write_text(failure, encoding='utf-8')
        (args.output/'run_status.json').write_text(json.dumps(dict(status='failed', error=failure)), encoding='utf-8')
    finally:
        if env is not None: env.close()
        app.close(wait_for_replicator=False, exit_code=1 if failure else 0)


if __name__ == '__main__': main()
