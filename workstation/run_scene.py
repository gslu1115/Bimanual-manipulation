"""Build and run the dual-Panda sorting environment using Isaac Sim's Python."""
import argparse
import json
import sys
import traceback
import faulthandler
import os
import time
from pathlib import Path

from task_logic import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('outputs')/'seed_006')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--keep-open', action='store_true')
    parser.add_argument('--no-sensors', action='store_true')
    parser.add_argument('--full-app', action='store_true', help='Use the standard full Python experience instead of sorting.kit')
    parser.add_argument('--robot-usd', help='Override the official Franka USD reference with a local USD/URL')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--conveyor-test', action='store_true', help='Three preloaded boxes: isolated physical conveyor test, no PnP claim')
    parser.add_argument('--single-pnp', action='store_true', help='One upright carton, physical gripper and Lula IK; ground-truth baseline')
    parser.add_argument('--dual-handover', action='store_true', help='Fixed inverted carton: physical dual-arm handover and upright placement')
    parser.add_argument('--record-video', action='store_true', help='Record dual-handover RGB cameras to MP4')
    args = parser.parse_args()
    if args.record_video and (args.no_sensors or not args.dual_handover):
        parser.error('--record-video requires --dual-handover and enabled sensors')
    c = json.loads(args.config.read_text(encoding='utf-8'))
    if args.seed is not None: c['seed'] = args.seed
    if sum([args.single_pnp, args.conveyor_test, args.dual_handover]) > 1: parser.error('Select only one task mode')
    if args.dual_handover:
        if 'dual_handover' not in c: parser.error('Configuration needs dual_handover parameters')
        c['boxes']['count'] = 1
    if args.single_pnp:
        if 'single_pick_place' not in c: parser.error('Configuration needs single_pick_place parameters')
        c['boxes']['count'] = 1
    elif args.conveyor_test:
        c['boxes']['count'] = 3
    c['run_mode'] = 'dual_handover' if args.dual_handover else 'single_pick_place' if args.single_pnp else 'conveyor_fixture' if args.conveyor_test else 'clutter_release'
    validate(c)
    if args.headless and args.keep_open: parser.error('--keep-open requires a GUI')
    if args.validate_only:
        print(f"Configuration valid: {c['run_mode']}, {c['boxes']['count']} cartons, 2 Panda arms, 3 virtual conveyor targets.")
        return
    from isaacsim import SimulationApp
    faulthandler.enable()
    faulthandler.dump_traceback_later(600 if (args.single_pnp or args.dual_handover) else 120, repeat=True)
    isaac_root = Path(os.environ.get('ISAAC_PATH', 'D:/isaacsim'))
    extra = ['--ext-folder', str(isaac_root/'extscache'), '--ext-folder', str(isaac_root/'extsDeprecated')]
    app = SimulationApp({'headless': args.headless, 'width': 960, 'height': 600,
                         'renderer': 'RaytracedLighting', 'anti_aliasing': 2,
                         'multi_gpu': False, 'sync_loads': True, 'extra_args': extra},
                        experience='' if args.full_app else str(Path(__file__).with_name('sorting.kit').resolve()))
    env = None
    failure = None
    try:
        from environment import SortingEnvironment
        env = SortingEnvironment(app, c, args.output.resolve(), args.robot_usd, not args.no_sensors, args.conveyor_test, args.single_pnp, args.dual_handover)
        env.record_video = args.record_video
        env.start()
        if args.conveyor_test:
            for _ in range(round(c['conveyor']['timeout_s']/c['physics']['dt'])+240):
                env.step()
                if env.conveyor.completed_batches or env.conveyor.state == 'FAULT': break
            if env.conveyor.completed_batches != 1:
                env._write_json('run_report.json', env.report())
                raise RuntimeError('Physical conveyor test did not clear one full batch')
        else:
            report = env.settle()
            if not report['settled'] or not report['inside_upstream_tray'] or not report['above_table']:
                env.capture('failed_settle')
                raise RuntimeError('Release episode did not settle inside the tray. Inspect settle_report.json.')
        sensor_report = env.capture()
        if args.single_pnp:
            from single_pick_place import SinglePickPlace
            result = SinglePickPlace(env).run()
            if not result['success']: raise RuntimeError('Single-box pick and place failed; see task_report.json')
        if args.dual_handover:
            from dual_handover import DualHandover
            result = DualHandover(env).run()
            if not result['success']: raise RuntimeError('Dual handover failed; see task_report.json')
        env._write_json('run_report.json', env.report())
        env._write_json('run_status.json', {'status': 'passed', 'mode': c['run_mode']})
        (args.output/'failure.txt').unlink(missing_ok=True)
        print('[ready] Outputs: '+str(env.output), flush=True)
        print('[sensors] '+json.dumps(sensor_report), flush=True)
        faulthandler.cancel_dump_traceback_later()
        while args.keep_open and app.is_running():
            start = time.perf_counter()
            if env.timeline.is_playing():
                env.step()
            else:
                app.update()
            time.sleep(max(0., c['physics']['dt']-(time.perf_counter()-start)))
    except BaseException:
        failure = traceback.format_exc()
        print(failure, file=sys.stderr, flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'failure.txt').write_text(failure, encoding='utf-8')
        (args.output/'run_status.json').write_text(json.dumps({'status': 'failed', 'error': failure}), encoding='utf-8')
    finally:
        faulthandler.cancel_dump_traceback_later()
        if env is not None: env.close()
        app.close(wait_for_replicator=False, exit_code=1 if failure else 0)
    if failure:
        raise RuntimeError('Simulation failed; see failure.txt')


if __name__ == '__main__':
    main()
