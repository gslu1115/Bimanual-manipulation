"""Offline development ledger. Never imported by the online policy."""
import hashlib
import json
import re
from pathlib import Path


def main(evidence_name='visual_continuation_20261008.json',start_run=1):
    if Path(evidence_name).name!=evidence_name or not evidence_name.endswith('.json'):
        raise ValueError('Evidence name must be a JSON basename')
    if start_run<1:raise ValueError('Starting trial must be positive')
    root=Path(__file__).resolve().parents[2]
    evidence=root/'evidence'/evidence_name
    if evidence.exists():raise FileExistsError('Historical evidence exists; select a new --evidence-name')
    output=root/'outputs/visual_continuation_20261008'
    initial=json.loads((output/'initial_hashes.json').read_text(encoding='utf-8-sig'))
    missing=[name for name in initial if not (root/name).is_file()]
    changed=[name for name,sha in initial.items() if (root/name).is_file()
             and hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha]
    protected=[name for name in initial if name.startswith(('workstation/observations/',
        'workstation/baselines/','config/','evidence/'))]
    preservation=dict(initial_files=len(initial),missing=missing,changed=changed,
                      protected_files=len(protected),protected_changed=[n for n in changed if n in protected])
    preservation['intentional_calibration_fix']=dict(
        source='workstation/observations/camera_geometry.py',
        backup='outputs/visual_continuation_20261008/camera_geometry_before_pixel_centre_fix.py',
        change='Replicator raster corner -> integer pixel centre cx/cy -0.5; existing API and three-camera architecture retained',
        measured_evidence='outputs/visual_continuation_20261008/pixel_centre_offset_audit.json')
    (output/'preservation_audit.json').write_text(json.dumps(preservation,ensure_ascii=False,indent=2),encoding='utf-8')
    trials=[];pending=[]
    saved_config=json.loads((root/'config/scene.json').read_text(encoding='utf-8'))
    for folder in sorted((root/'outputs').glob('visual_space_pnp_20261008_*')):
        if int(folder.name.rsplit('_',1)[-1])<start_run:continue
        report=folder/'visual_task_report.json'
        if not report.exists():pending.append(folder.name);continue
        result=json.loads(report.read_text(encoding='utf-8'))
        config=json.loads((folder/'config_used.json').read_text(encoding='utf-8'))
        trials.append(dict(run=str(folder.relative_to(root)),success=result['success'],
            failure_reason=result.get('failure_reason'),phases=[event['phase'] for event in result['events']],
            max_visual_lift_m=result['max_visual_lift_m'],held_on_exit=result['held_on_exit'],
            closed_surface_support=result.get('closed_surface_support'),
            release_geometry=result.get('release_geometry'),
            released_collision_estimate=result.get('released_collision_estimate'),
            failure_workspace=result.get('failure_path_check',{}).get('workspace'),
            config_used=str((folder/'config_used.json').relative_to(root)),
            source_hashes=str((folder/'online_source_hashes.json').relative_to(root)),
            report=str(report.relative_to(root)),
            observer_preparation=result.get('observer_preparation'),
            post_observer_task_check=result.get('post_observer_task_check'),
            controlled_conditions={key:config.get(key) for key in (
                'diagnostic_robot_stow_home','diagnostic_preposition_observer',
                'diagnostic_depth_backboard','diagnostic_scene_view','diagnostic_right_wrist_view',
                'diagnostic_scene_height_m','diagnostic_wrist_focal_length_mm',
                'diagnostic_optical_depth_error_margin_m')},
            camera_configuration_matches_saved=config['cameras']==saved_config['cameras'],
            optical_depth_calibration_gate=json.loads((folder/'optical_depth_calibration_gate.json').read_text(encoding='utf-8'))
                if (folder/'optical_depth_calibration_gate.json').is_file() else None,
            evaluation_scope='post-actor offline only; never a policy input'))
    (output/'space_trials.json').write_text(json.dumps(dict(trials=trials,pending=pending,
        count_is_success_rate=False,reason='development changes to source/optics/fixtures between trials'),
        ensure_ascii=False,indent=2),encoding='utf-8')
    checks={}
    for name,pattern in (('isolated','logic_*.txt'),('sdk','sdk_logic_*.txt')):
        paths=list(output.glob(pattern))
        if not paths:continue
        path=max(paths,key=lambda p:p.stat().st_mtime);body=path.read_text(encoding='utf-8-sig')
        count=re.findall(r'Ran (\d+) tests?',body)
        checks[name]=dict(log=str(path.relative_to(root)),count=int(count[-1]) if count else None,
            passed=bool(re.search(r'^OK\s*$',body,re.MULTILINE)),
            command='unittest.defaultTestLoader.discover("tests")',scope='logic only, not physical acceptance')
    suites={}
    for name,filename in (('fixed_repeat','frozen_repeat_results56_57.json'),
                          ('independent_upright','independent_upright_results101_103.json')):
        path=output/filename
        if path.is_file():
            suites[name]=dict(report=str(path.relative_to(root)),
                results=json.loads(path.read_text(encoding='utf-8')))
    summary=dict(schema='visual_continuation_development_v1',date='2026-10-08',logic_checks=checks,
        initial_git_status='outputs/visual_continuation_20261008/initial_git_status.txt',
        preservation=preservation,resources=dict(gpu='RTX 4060 Laptop 8188 MiB',driver='560.94',ram_total_gib=31.7967,
            isaac='6.1.0-rc.26',isaac_python='3.12.13',isaac_numpy='2.3.1',model_python='3.12.13',
            model_numpy='2.2.6',torch='2.6.0+cu124',ultralytics='8.3.253',cuda_available=True,
            inference_isolated=True,core_dependencies_modified=False),
        model=dict(family='project fine-tuned YOLO11n-seg',
            weights='outputs/carton_seg_motion_training_20261007/fit/weights/best.pt',
            sha256='402e0c8e63005e408624ce85544a5ae1da5784c72dd67b4b67ed6d999fa35f11',confidence=.4,
            historical_selection='evidence/model_progress_20261008.json',
            zero_shot_yoloe_accepted=False,grasp_source='geometric templates; user selected; no GraspGen inference result'),
        historical=dict(earlier_fixed_visual_repeats='2/2; before unknown-space constraints; not current acceptance',
            evidence='outputs/model_integration_20261007/fixed_repeat_results_20261008.json'),
        current=dict(completed_trial_records=len(trials),complete_success_records=sum(t['success'] for t in trials),
            first_recorded_trial=start_run,count_is_success_rate=False,
            ledger_includes_separately_reported_frozen_repeats=True,
            pending=pending,trials=trials,acceptance='outputs/visual_continuation_20261008/stage_acceptance.json',
            separate_acceptance_suites=suites,
            same_version_random_success_rate_established=False),
        implementation=dict(observed_space='10mm cells; 2.5mm boundary samples; calibrated valid optical depth; unknown blocks',
            robot_model='official Panda convex bodies for image-time self filtering;36 onward removes whole-arm convex boundary query;39 onward removes finger-shell visibility; keeps Lula sphere collisions/observed obstacles and 3mm TCP corridor/held-box unknown gates',
            whole_arm_unknown_space_proven=False,
            finger_shell_visibility_required=False,
            future_local_visibility='38 onward explicitly pending in preflight; runtime requires fresh frames',
            held_geometry='40 onward latched visual box-to-TCP relation shared with holding monitor; not a physical attachment',
            held_space='42 onward oriented metric cuboid with3mm padding; not its enclosing AABB; unknown interior remains checked',
            upright_tcp_height_m=.004,
            joint_planner='bounded local RRT-Connect; every edge checked; exhaustion does not prove unreachable',
            reobserve='current passive fresh frames only; historical empty wrist lifts/rotations/prefixes retained as diagnostics',
            current_camera_constraint='saved config camera installation/optics preserved; dedicated active view moves disabled by default; old experiments are separate',
            occlusion_memory='previous valid optical rays; current supported robot/payload foreground or measured farther background in all9 pixels; ordinary5s expiry; unexplained foreground/other-object motion veto; never creates unseen free space',
            payload_history='43 onward up to4 pre-action optical-ray anchors <=30s, only with current supported payload; other-object envelopes/unexplained foreground veto; planning payload not physical attachment',
            mixed_depth_footprint='45 onward erode union of raw robot/payload support; all9 pixels required; raw masks saved44 onward',
            payload_mask_holes='46 onward self filter only: enclosed <=9pixel holes or1pixel-near-mask dropouts, with measured surface <=3mm; original masks and holding evidence unchanged',
            payload_lifecycle='48 onward explicit and periodic observations both revalidate current holding support, no old-frame reuse;49 onward transferred/lowered frames saved before descent/release, current support required for carrying motion',
            cross_view_payload_filter='50 onward another CURRENT supported holding view permits unlabelled measured pixels within3mm of the payload surface as self depth only; original masks/holding evidence unchanged; old free rays still mandatory',
            payload_proxy='current confirmed oriented payload supersedes only its own clipped visual proxy; stale geometry revoked; neighbours and unseen swept volume retained',
            current_payload_uncertainty='latched visual pose uncertainty >=3mm, <=8mm plus existing2mm mapping allowance; current modelled occupied volume only, not a free-space label',
            held_collision_uncertainty='same latched visual uncertainty >=3mm; invalid/unbounded values reject rather than disappear when replacing clipped proxy',
            measured_vertical_carry='closed image-time FK TCP plus180mm lift; unchanged yaw purevertical carrying retains measured quaternion, existing full sampled path checks still mandatory',
            near_fixed_surface='measured fixed-face match2mm just BEHIND query in optical band can explain a current pixel; all9 valid pixels and an independent older free ray still required; never marks unseen volume free',
            nominal_body_release_gap_m=.010,nominal_place_tcp_clearance_m=.014,
            release_geometry='drop bound<=20mm from current supported visual geometry and image-timeFK/knownsupport; after opening translation fitted from current top/signedfaces/extents, lateral consistency<=8mm, pre-release centre remains anchor rather than following arm unloading',
            release_grip='54 onward <=1mm finger increments checked with currently supported held oriented body; missing/stale support or model collision refuses before drive; not a contact exemption for the palm',
            empty_retreat='55 onward finger contact permission separated from TCP approach gate; actual approach gate retained; released RGB-D geometry replaces its clipped proxy in collision and workspace models; empty retreat uses sphere collisions/observed obstacles but unknown-space safety is explicitly NOT proved',
            elevated_posture='short elevated footprint without measured height remains UNKNOWN with SIDE/UP/INV hypotheses; uncertain centres do not produce observed velocity',
            observer_preposition='optional checked empty wrist staging; measured camera pose and its own depth ray; fresh full-task check required',
            sparse_convex_query='historical/offline only; not invoked by current whole-arm execution',
            task_loop='visual-sort; shared tracks, assigned slots, bounded failures/actions; physical multi-box unaccepted'),
        boundaries=dict(truth_as_online_input=False,box_attachment=False,box_pose_teleport=False,
            original_three_cameras_rebuilt=False,continuous_collision_proof=False,
            empty_retreat_unknown_space_proven=False,
            strict_exposure_ids_verified=False,default_scene_config_changed=False,
            depth_backboard='optional controlled diagnostic fixture; physically rendered and colliding; not synthetic depth'),
        calibration=dict(pixel_centre='integer image indices; rendered optical centre minus0.5 pixel',
            validation='outputs/visual_continuation_20261008/pixel_centre_offset_audit.json',
            historical_frames_rewritten=False,default_depth_free_margin_m=.003,
            explicit_margin_experiments='33/34 used1mm only after saved three-camera static-plane gate;35 restores saved configuration/default3mm',
            scope='actual SDK pinhole static-plane frames; not real camera calibration'),
        unavailable_or_unaccepted=['GraspGen Linux service and Panda TCP calibration',
            'robust inverted semantic evidence','visual SideAdjustment and DualArmFlip physical skills',
            'randomized single-box feasible domain','separated multi-box execution','clutter/nine-box sorting',
            'continuous collision guarantee','hardware hand-eye calibration'],
        sources=['https://docs.ultralytics.com/models/yoloe/',
            'https://github.com/NVlabs/GraspGen',
            'https://docs.isaacsim.omniverse.nvidia.com/6.1.0/manipulators/manipulators_lula_rrt.html',
            'https://github.com/moveit/moveit2/blob/main/moveit_ros/planning/planning_scene_monitor/src/planning_scene_monitor.cpp',
            'https://github.com/frankarobotics/franka_ros/blob/develop/franka_description/doc/index.rst'])
    evidence.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(trials=len(trials),successes=sum(t['success'] for t in trials),pending=pending,
                         preservation=preservation),ensure_ascii=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence-name',default='visual_continuation_20261008.json')
    parser.add_argument('--start-run',type=int,default=1)
    args=parser.parse_args()
    main(args.evidence_name,args.start_run)
