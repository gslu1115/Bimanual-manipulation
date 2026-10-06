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
MODES = ("scene", "single-pnp", "dual-handover", "conveyor-check", "sensor-check")


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
    return parser


def load_config(args, parser):
    if args.headless and args.keep_open:
        parser.error("--keep-open requires a GUI")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    if args.mode in ("single-pnp", "dual-handover", "sensor-check"):
        config["boxes"]["count"] = 1
    elif args.mode == "conveyor-check":
        config["boxes"]["count"] = 3
    config["run_mode"] = args.mode
    if args.save_images:
        config.setdefault("camera_system", {})["save_images"] = True
    if args.camera_recording:
        config.setdefault("camera_system", {})["recording_enabled"] = True
    settings = camera_settings(config)
    if args.record_video and args.mode != "dual-handover":
        parser.error("--record-video requires --mode dual-handover")
    if (args.mode == "sensor-check" or args.camera_recording or args.record_video) and (
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
    config = load_config(args, parser)
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
        env = SortingEnvironment(app, config, output, args.robot_usd,
            sensors=not args.no_sensors, conveyor_test=args.mode == "conveyor-check",
            single_pnp=args.mode in ("single-pnp", "sensor-check"),
            dual_handover=args.mode == "dual-handover")
        env.record_video = args.record_video
        env.start()
        if args.mode == "sensor-check":
            from workstation.diagnostics.sensor_check import check
            check(env)
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
