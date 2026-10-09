"""Unified command-line entry point for the simulated workcell."""
from __future__ import annotations
import argparse
from datetime import datetime
import faulthandler
import json
import os
from pathlib import Path
import sys
import time
import traceback
from workstation.task_logic import validate
from workstation.observations.camera_config import camera_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("scene", "single-pnp", "dual-handover", "conveyor-check", "sensor-check", "vision-check", "visual-pnp", "visual-sort",
         "model-worker", "model-prepare", "model-eval", "model-capture", "model-train")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="scene")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT/"config"/"scene.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--no-sensors", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--camera-recording", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--robot-usd")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT/'models/yoloe-11s-cardboard.pt')
    parser.add_argument("--prompt-mode", choices=('text','visual','trained'), default='text')
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--segmenter", choices=('geometry','yoloe','yolo-seg'), default='geometry')
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--fixture", choices=('upright','side','inverted','separated','clutter'),default='upright')
    parser.add_argument('--capture-seeds',help='model-capture only: comma-separated independent initial scenes')
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--teacher-capture',action='store_true',help='Offline truth-baseline supervision, model-capture upright only')
    parser.add_argument('--visual-fixture',choices=('upright','side','inverted','separated-upright','separated','clutter'),
                        help='Explicit diagnostic INITIAL condition for visual-pnp; seed then changes XY/yaw')
    parser.add_argument('--scene-focal-length-mm',type=float,
                        help='Explicit calibrated scene-camera optics experiment; original config is preserved')
    parser.add_argument('--wrist-focal-length-mm',type=float,
                        help='INITIAL optics experiment for both existing wrist cameras, 10..30mm')
    parser.add_argument('--max-items',type=int,default=3,help='visual-sort only: requested number of visually verified slots')
    parser.add_argument('--scene-view',choices=('front','rear','overhead','west'),default='front',
                        help='Explicit INITIAL camera installation experiment; no runtime camera pose writes')
    parser.add_argument('--right-wrist-view',choices=('default','cross-left','cross-left-offset'),default='default',
                        help='INITIAL nominal wrist mounting experiment for complementary left-arm workspace coverage')
    parser.add_argument('--scene-look-height-m',type=float,
                        help='INITIAL scene-camera aim-height experiment, 0.8..1.2m; same camera and fresh calibration')
    parser.add_argument('--scene-height-m',type=float,
                        help='INITIAL scene-camera installation height experiment, 1.1..2.5m')
    parser.add_argument('--optical-depth-margin-mm',type=float,
                        help='Explicit calibrated optical-depth error margin, 1..10mm; below3mm requires static-board diagnostic')
    parser.add_argument('--depth-backboard',action='store_true',
                        help='Explicit controlled depth acquisition experiment with a fixed colliding west background board')
    parser.add_argument('--robot-stow-home',action='store_true',
                        help='INITIAL robot observation pose outside bounded visual workspace; preserves original config')
    parser.add_argument('--preposition-observer',action='store_true',
                        help='Checked empty-other-wrist observation before the active arm enters its pregrasp pose')
    return parser


def load_config(args, parser):
    if args.headless and args.keep_open:
        parser.error("--keep-open requires a GUI")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.preposition_observer:
        if args.mode not in ('visual-pnp','visual-sort'):parser.error('--preposition-observer requires a visual skill')
        config['diagnostic_preposition_observer']=True
    if args.robot_stow_home:
        if args.mode not in ('visual-pnp','visual-sort','vision-check','model-capture'):
            parser.error('--robot-stow-home requires a visual diagnostic mode')
        # Initial setup only, shared by the existing scene initializer and HOME
        # contract. Official FK scan found no robot body (+10mm) in the bounded
        # observed volume. This is not a runtime joint-position setter.
        config['robot_home']=[0.,-.55,0.,-1.90,0.,1.35,.7853981633974483,.04,.04]
        config['diagnostic_robot_stow_home']=True
    if args.seed is not None:
        config["seed"] = args.seed
    if args.mode in ("single-pnp", "dual-handover", "sensor-check", "vision-check", "visual-pnp"):
        config["boxes"]["count"] = 1
    elif args.mode == "conveyor-check":
        config["boxes"]["count"] = 3
    elif args.mode == 'model-capture':
        config['boxes']['count'] = 9 if args.fixture == 'clutter' else 3 if args.fixture == 'separated' else 1
    config["run_mode"] = args.mode
    if args.depth_backboard:
        if args.mode not in ('visual-pnp','visual-sort','vision-check','model-capture'):
            parser.error('--depth-backboard requires a visual diagnostic mode')
        config['diagnostic_depth_backboard']=True
    if args.optical_depth_margin_mm is not None:
        if args.mode not in ('visual-pnp','visual-sort','vision-check'):
            parser.error('--optical-depth-margin-mm requires a visual diagnostic mode')
        if not 1<=args.optical_depth_margin_mm<=10:parser.error('Optical depth margin must be 1..10mm')
        if args.optical_depth_margin_mm<3 and not args.depth_backboard:
            parser.error('Below3mm requires --depth-backboard and separate saved calibration audit')
        config['diagnostic_optical_depth_error_margin_m']=args.optical_depth_margin_mm/1000.
    if args.visual_fixture:
        if args.mode not in ('visual-pnp','visual-sort','vision-check'): parser.error('--visual-fixture requires a visual mode')
        config['boxes']['count']=9 if args.visual_fixture == 'clutter' else 3 if args.visual_fixture.startswith('separated') else 1
        config['diagnostic_visual_fixture']=args.visual_fixture
    if args.scene_focal_length_mm is not None:
        if args.mode not in ('visual-pnp','visual-sort','vision-check','model-capture'):
            parser.error('--scene-focal-length-mm requires a visual diagnostic mode')
        if not 10 <= args.scene_focal_length_mm <= 50:parser.error('Scene focal length must be 10..50mm')
        next(cam for cam in config['cameras'] if cam['name']=='scene_camera')['focal_length_mm']=args.scene_focal_length_mm
    if args.wrist_focal_length_mm is not None:
        if args.mode not in ('visual-pnp','visual-sort','vision-check','model-capture'):
            parser.error('--wrist-focal-length-mm requires a visual diagnostic mode')
        if not 10<=args.wrist_focal_length_mm<=30:parser.error('Wrist focal length must be 10..30mm')
        for cam in config['cameras']:
            if cam['name'] in ('left_wrist_camera','right_wrist_camera'):
                cam['focal_length_mm']=args.wrist_focal_length_mm
        config['diagnostic_wrist_focal_length_mm']=args.wrist_focal_length_mm
    if args.scene_view!='front':
        if args.mode not in ('visual-pnp','visual-sort','vision-check','model-capture'):
            parser.error('--scene-view requires a visual diagnostic mode')
        camera=next(cam for cam in config['cameras'] if cam['name']=='scene_camera')
        camera['position']={'rear':[0.,.60,1.76],'overhead':[0.,.04,2.10],
                            'west':[-.85,-.10,1.70]}[args.scene_view]
        camera['look_at']=[0.,-.11,.85]
        config['diagnostic_scene_view']=args.scene_view
    if args.scene_height_m is not None:
        if args.mode not in ('visual-pnp','visual-sort','vision-check','model-capture'):
            parser.error('--scene-height-m requires a visual diagnostic mode')
        if not 1.1<=args.scene_height_m<=2.5:parser.error('Scene installation height must be 1.1..2.5m')
        next(cam for cam in config['cameras'] if cam['name']=='scene_camera')['position'][2]=args.scene_height_m
        config['diagnostic_scene_height_m']=args.scene_height_m
    if args.scene_look_height_m is not None:
        if args.mode not in ('visual-pnp','visual-sort','vision-check','model-capture'):
            parser.error('--scene-look-height-m requires a visual diagnostic mode')
        if not .8<=args.scene_look_height_m<=1.2:parser.error('Scene aim height must be 0.8..1.2m')
        camera=next(cam for cam in config['cameras'] if cam['name']=='scene_camera')
        camera['look_at'][2]=args.scene_look_height_m
        config['diagnostic_scene_look_height_m']=args.scene_look_height_m
    if args.right_wrist_view!='default':
        if args.mode not in ('visual-pnp','visual-sort','vision-check','model-capture'):
            parser.error('--right-wrist-view requires a visual diagnostic mode')
        camera=next(cam for cam in config['cameras'] if cam['name']=='right_wrist_camera')
        # Nominal fixed hand-eye rotation, chosen using the known home robot
        # geometry and workcell volume; no online carton pose or identity input.
        # The existing adapter still computes each frame's transform from FK.
        camera['mount_orientation_wxyz']=[.10387994761833542,-.09840339722240588,
                                         .6759348358454595,-.7229369444069609]
        if args.right_wrist_view=='cross-left-offset':
            # A 60mm nominal side bracket avoids looking through the right
            # palm. It is an INITIAL installation experiment, not a runtime
            # world-pose setter or an extra camera. Existing FK calibration
            # uses this same declared hand-eye transform at each image time.
            camera['mount_translation']=[.065,.060,.025]
            camera['mount_orientation_wxyz']=[.11690687965799523,-.1323562996133372,
                                             .7343218666450804,-.6554280949182605]
        config['diagnostic_right_wrist_view']=args.right_wrist_view
    if args.mode=='visual-sort' and (args.visual_fixture is None or not 1<=args.max_items<=3):
        parser.error('visual-sort requires explicit --visual-fixture and --max-items 1..3')
    if args.save_images:
        config.setdefault("camera_system", {})["save_images"] = True
    if args.camera_recording:
        config.setdefault("camera_system", {})["recording_enabled"] = True
    settings = camera_settings(config)
    if args.record_video and args.mode != "dual-handover":
        parser.error("--record-video requires --mode dual-handover")
    if (args.mode in ("sensor-check", "vision-check", "visual-pnp", "visual-sort", "model-capture") or args.camera_recording or args.record_video) and (
            args.no_sensors or not settings["enabled"]):
        parser.error("Requested sensor operation requires enabled cameras")
    if args.record_video and any(not cam.get("enabled", True) for cam in config["cameras"]):
        parser.error("Three-camera recording requires all three cameras")
    validate(config)
    if args.record_video and len({tuple(cam["resolution"]) for cam in config["cameras"]}) != 1:
        parser.error("Three-camera video requires equal image resolutions")
    return config


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.confidence is None:
        args.confidence=.4 if args.prompt_mode == 'trained' or args.segmenter == 'yolo-seg' else .15
    if not 0 < args.confidence < 1: parser.error('--confidence must be between 0 and 1')
    if args.teacher_capture and (args.mode != 'model-capture' or args.fixture != 'upright'):
        parser.error('--teacher-capture requires model-capture --fixture upright')
    if args.mode in ('model-worker','model-prepare','model-eval','model-train'):
        if args.mode == 'model-worker':
            from workstation.models.yoloe_worker import serve
            return serve(args)
        if args.mode == 'model-prepare':
            from workstation.models.yoloe_worker import prepare
            return prepare(args)
        if args.mode == 'model-train':
            from workstation.diagnostics.model_training import train
            return train(args)
        from workstation.diagnostics.model_evaluation import evaluate
        return evaluate(args)
    config = load_config(args, parser)
    capture_seeds=[]
    if args.capture_seeds:
        if args.mode != 'model-capture': parser.error('--capture-seeds requires model-capture')
        capture_seeds=[int(value) for value in args.capture_seeds.split(',')]
        if len(set(capture_seeds)) != len(capture_seeds): parser.error('Capture seeds must be unique')
        config['seed']=capture_seeds[0]
    if args.validate_only:
        print(f"Configuration valid: {args.mode}, {config['boxes']['count']} cartons, "
              "2 Panda arms, 3 formal cameras.")
        return
    output = (args.output or PROJECT_ROOT/"outputs"/(
        args.mode+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    # Isaac imports must follow SimulationApp initialization.
    from isaacsim import SimulationApp
    root = Path(os.environ.get("ISAAC_PATH", "D:/isaacsim"))
    app = SimulationApp({"headless": args.headless, "width": 960, "height": 600,
        "renderer": "RaytracedLighting", "anti_aliasing": 2, "multi_gpu": False,
        "sync_loads": True, "extra_args": ["--ext-folder", str(root/"extscache"),
        "--ext-folder", str(root/"extsDeprecated")]},
        experience=str(PROJECT_ROOT/"config"/"sorting.kit"))
    env, failure = None, None
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, repeat=True)
    try:
        from workstation.simulation.environment import SortingEnvironment
        env_output=output/('seed_'+str(config['seed'])) if capture_seeds else output
        env = SortingEnvironment(app, config, env_output, args.robot_usd,
            sensors=not args.no_sensors, conveyor_test=args.mode == "conveyor-check",
            single_pnp=args.mode in ("single-pnp", "sensor-check", "vision-check", "visual-pnp"),
            dual_handover=args.mode == "dual-handover",
            visual_fixture=args.fixture if args.mode == 'model-capture' else args.visual_fixture)
        env.record_video = args.record_video
        env.start()
        if args.mode == "sensor-check":
            from workstation.diagnostics.sensor_check import check
            check(env)
        elif args.mode == "vision-check":
            from workstation.diagnostics.vision_check import check
            if args.segmenter in ('yoloe','yolo-seg'):
                from workstation.models.yoloe_client import YOLOESegmenter
                with YOLOESegmenter('trained' if args.segmenter == 'yolo-seg' else args.prompt_mode,args.weights,args.reference,env.output/'model_worker.log',
                                    confidence=args.confidence) as segmenter:
                    check(env,segmenter)
            else: check(env)
        elif args.mode in ("visual-pnp","visual-sort"):
            from workstation.diagnostics.visual_run import run
            if args.segmenter in ('yoloe','yolo-seg'):
                from workstation.models.yoloe_client import YOLOESegmenter
                with YOLOESegmenter('trained' if args.segmenter == 'yolo-seg' else args.prompt_mode,args.weights,args.reference,env.output/'model_worker.log',
                                    confidence=args.confidence) as segmenter:
                    run(env,segmenter,max_items=args.max_items if args.mode=='visual-sort' else None)
            else: run(env,max_items=args.max_items if args.mode=='visual-sort' else None)
        elif args.mode == 'model-capture':
            from workstation.diagnostics.model_capture import capture, teacher_capture
            collect=teacher_capture if args.teacher_capture else capture
            collect(env,args.fixture)
            for seed in capture_seeds[1:]:
                env._write_json('run_status.json',{'status':'passed','mode':args.mode})
                env.close()
                for _ in range(5): app.update()
                config['seed']=seed
                env=SortingEnvironment(app,config,output/('seed_'+str(seed)),args.robot_usd,
                                       sensors=True,visual_fixture=args.fixture)
                env.start()
                collect(env,args.fixture)
        elif args.mode == "conveyor-check":
            for _ in range(round(config["conveyor"]["timeout_s"]/config["physics"]["dt"])+240):
                env.step()
                if env.conveyor.completed_batches or env.conveyor.state == "FAULT":
                    break
            if env.conveyor.completed_batches != 1:
                raise RuntimeError("Conveyor did not clear one full batch")
        else:
            settled = env.settle()
            if not all(settled[key] for key in ("settled", "inside_upstream_tray", "above_table")):
                raise RuntimeError("Cartons did not settle inside the tray")
            env.capture("settled")
            if args.mode == "single-pnp":
                from workstation.baselines.single_pick_place import SinglePickPlace
                if not SinglePickPlace(env).run()["success"]:
                    raise RuntimeError("Fixed single-box baseline failed")
            elif args.mode == "dual-handover":
                from workstation.baselines.dual_handover import DualHandover
                if not DualHandover(env).run()["success"]:
                    raise RuntimeError("Fixed dual-handover baseline failed")
        if args.mode not in ("single-pnp", "dual-handover"):
            env.export_snapshot("final.usda")
        env._write_json("run_report.json", env.report())
        env._write_json("run_status.json", {"status": "passed", "mode": args.mode})
        print("[ready] Outputs: "+str(output), flush=True)
        faulthandler.cancel_dump_traceback_later()
        while args.keep_open and app.is_running():
            started = time.perf_counter()
            if env.timeline.is_playing():
                env.step()
            else:
                app.update()
            time.sleep(max(0., config["physics"]["dt"]-(time.perf_counter()-started)))
    except BaseException:
        failure = traceback.format_exc()
        print(failure, file=sys.stderr, flush=True)
        output.mkdir(parents=True, exist_ok=True)
        (output/"failure.txt").write_text(failure, encoding="utf-8")
        (output/"run_status.json").write_text(json.dumps(
            {"status": "failed", "mode": args.mode, "error": failure}), encoding="utf-8")
    finally:
        faulthandler.cancel_dump_traceback_later()
        if env is not None:
            env.close()
        app.close(wait_for_replicator=False, exit_code=1 if failure else 0)
    if failure:
        raise RuntimeError("Simulation failed; see "+str(output/"failure.txt"))
