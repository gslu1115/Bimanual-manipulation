"""Shared USD mounts, Replicator observations and frame recording.

Import after SimulationApp. Wrist mounts inherit the PHYSICAL hand transform;
there are no camera world-pose setters, rigid-body constraints or colliders.
"""
import math
import subprocess
import time

import numpy as np
import omni.timeline
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.rendering_manager import RenderingManager

from workstation.observations.camera_config import FORMAL_CAMERAS, ANNOTATIONS, camera_settings
from workstation.observations.camera_geometry import calibration_from_params, intrinsics_from_params


class CameraSystem:
    def __init__(self, env, enabled=True):
        self.env = env
        self.settings = camera_settings(env.c)
        self.sensors, self.cameras, self.hand_paths = {}, {}, {}
        self._closed = False
        self.last_metadata = {}
        self.render_frame_index = 0
        self.rendered_tick, self.rendered_time = 0, 0.
        self.rendered_joint_positions = None
        self.rendered_robot_states = None
        self.metrics = dict(render_calls=0, render_wall_s=0., data_reads=0, data_read_wall_s=0.,
                            saved_frames=0, save_wall_s=0., gpu_memory=[])
        self._gpu_sample('before_render_products')
        active = enabled and self.settings['enabled']
        if active:
            import omni.replicator.core as rep
        for cfg in env.c['cameras']:
            name = cfg['name']
            role = 'formal'
            path, mount, hand = self._create_mount(cfg)
            cam = UsdGeom.Camera.Define(env.stage, path)
            if cfg.get('type', 'fixed') != 'wrist':
                if 'orientation_wxyz' in cfg:
                    cam.AddTranslateOp().Set(Gf.Vec3d(*cfg['position']))
                    q = cfg['orientation_wxyz']
                    cam.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
                else:
                    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*cfg['position']),
                        Gf.Vec3d(*cfg['look_at']), Gf.Vec3d(*cfg['up'])).GetInverse())
            # Wrist Camera is identity under its mount; all installation extrinsics
            # are authored on the mount in the panda_hand frame (scalar-first).
            cam.CreateFocalLengthAttr(cfg['focal_length_mm'])
            cam.CreateHorizontalApertureAttr(cfg['horizontal_aperture_mm'])
            w, h = cfg['resolution']
            aperture_y = cfg['horizontal_aperture_mm']*h/w
            cam.CreateVerticalApertureAttr(aperture_y)
            cam.CreateClippingRangeAttr(Gf.Vec2f(*cfg.get('clipping_range_m', [.01, 10.])))
            cam.CreateFStopAttr(0.)
            cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
            cam.GetPrim().CreateAttribute('cameraProjectionType', Sdf.ValueTypeNames.Token).Set('pinhole')
            sampling = active and cfg.get('enabled', True)
            descriptor = dict(prim_path=path, parent_prim_path=str(cam.GetPrim().GetParent().GetPath()),
                mount_prim_path=mount, hand_prim_path=hand, role=role, sampling_enabled=bool(sampling),
                resolution_wh=[w, h], focal_length_mm=cfg['focal_length_mm'],
                horizontal_aperture_mm=cfg['horizontal_aperture_mm'],
                horizontal_fov_deg=math.degrees(2*math.atan(cfg['horizontal_aperture_mm']/2/cfg['focal_length_mm'])),
                mount_translation_m=cfg.get('mount_translation'),
                mount_orientation_wxyz=cfg.get('mount_orientation_wxyz'))
            self.cameras[name] = dict(config=cfg, camera=cam, descriptor=descriptor)
            # Initial authored pose only. Saved frame calibration always comes
            # from the renderer, including Fabric-updated dynamic wrist poses.
            initial = calibration_from_params(dict(cameraViewTransform=np.asarray(
                cam.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()),
                cameraFocalLength=cfg['focal_length_mm'],
                cameraAperture=[cfg['horizontal_aperture_mm'], aperture_y]), [w, h])
            initial.update(descriptor, pose_source='authored_initial_usd_not_a_rendered_frame',
                           dynamic_world_pose=hand is not None)
            env._write_json(name+'_calibration.json', initial)
            if sampling:
                self._attach_sensor(name)
        self._timeline_subscription = env.timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event, order=-1000, name='Bimanual camera lifecycle')
        env._write_json('camera_rig.json', self.describe())
        print('[camera-rig] '+str({n: c['descriptor']['prim_path'] for n, c in self.cameras.items()}), flush=True)

    def _attach_sensor(self, name):
        import omni.replicator.core as rep
        camera = self.cameras[name]
        cfg = camera['config']
        product = rep.create.render_product(camera['descriptor']['prim_path'], tuple(cfg['resolution']))
        annotators = {}
        for label in dict.fromkeys(list(cfg.get('annotations', ANNOTATIONS))+['camera_params']):
            ann = rep.AnnotatorRegistry.get_annotator(label)
            ann.attach([product])
            annotators[label] = ann
        self.sensors[name] = dict(product=product, annotators=annotators)

    def _on_timeline_event(self, event):
        if self._closed:
            return
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._release_sensors()
            self.last_metadata.clear()
            self.rendered_robot_states = None
            self.rendered_joint_positions = None
        elif event.type == int(omni.timeline.TimelineEventType.PLAY):
            for name, camera in self.cameras.items():
                if camera['descriptor']['sampling_enabled'] and name not in self.sensors:
                    self._attach_sensor(name)

    def _hand(self, robot):
        if robot not in self.hand_paths:
            root = self.env.stage.GetPrimAtPath('/World/Robots/'+robot)
            paths = [str(p.GetPath()) for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                     if p.GetName() == 'panda_hand' and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(paths) != 1:
                raise RuntimeError(f'Expected one physical panda_hand for {robot}, found {paths}')
            prim = self.env.stage.GetPrimAtPath(paths[0])
            if prim.IsInstanceProxy():
                raise RuntimeError(f'Cannot author wrist mount under instance proxy: {paths[0]}')
            self.hand_paths[robot] = paths[0]
        return self.hand_paths[robot]

    def _create_mount(self, cfg):
        name = cfg['name']
        if cfg.get('type', 'fixed') == 'wrist':
            hand = self._hand(cfg['robot'])
            mount_path = hand+'/'+name.replace('_camera', '_camera_mount')
            mount = UsdGeom.Xform.Define(self.env.stage, mount_path)
            mount.AddTranslateOp().Set(Gf.Vec3d(*cfg['mount_translation']))
            q = cfg['mount_orientation_wxyz']
            mount.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
            mount.GetPrim().SetCustomDataByKey('sensor_mount_frame', 'physical panda_hand; wxyz; USD optical -Z')
            return mount_path+'/'+name, mount_path, hand
        UsdGeom.Xform.Define(self.env.stage, '/World/perception_rig')
        return '/World/perception_rig/'+name, None, None

    def render(self):
        # Auto physics stepping is disabled. Sample measured joints at the physics
        # state that will be rendered, then keep them with this image tick.
        sampled_tick, sampled_time = self.env.ticks, self.env.time
        sampled_states = self.env.observe_robot_states()
        started = time.perf_counter()
        RenderingManager.render()
        self.env.app.update()
        self.metrics['render_wall_s'] += time.perf_counter()-started
        self.metrics['render_calls'] += 1
        self.render_frame_index += 1
        self.rendered_tick, self.rendered_time = sampled_tick, sampled_time
        self.rendered_robot_states = sampled_states
        # Compatibility diagnostic snapshot in each articulation's native DOF order.
        from workstation.observations.observation_packet import PANDA_ARM_JOINT_NAMES, PANDA_FINGER_JOINT_NAMES
        names = PANDA_ARM_JOINT_NAMES + PANDA_FINGER_JOINT_NAMES
        self.rendered_joint_positions = {
            name: state.qpos[[names.index(joint) for joint in self.env.arms[name].dof_names]]
            for name, state in sampled_states.items()}

    def observe(self, refresh=False, tolerate_errors=False, include_privileged=True):
        if not self.sensors:
            return dict(cameras={}, camera_errors={})
        if refresh:
            for _ in range(self.settings['warmup_render_frames']): self.render()
        started = time.perf_counter()
        frames, errors = {}, {}
        for name, sensor in self.sensors.items():
            try:
                cfg = self.cameras[name]['config']
                w, h = cfg['resolution']
                anns = sensor['annotators']
                params = anns['camera_params'].get_data()
                if not isinstance(params, dict):
                    raise RuntimeError(f'Camera {name} produced no rendered camera parameters')
                if list(np.asarray(params['renderProductResolution']).astype(int)) != [w, h]:
                    raise RuntimeError(f'Camera {name} render-product resolution mismatch')
                if include_privileged:
                    metadata = calibration_from_params(params, [w, h])
                    metadata.update(self.cameras[name]['descriptor'], simulation_time_s=self.rendered_time,
                                    physics_frame_index=self.rendered_tick, render_frame_index=self.render_frame_index)
                    k = metadata['K']
                else:
                    metadata = None
                    k = intrinsics_from_params(params, [w, h]).tolist()
                frame = dict(rgb=None, depth=None, intrinsics=k)
                if include_privileged:
                    frame.update(instance_segmentation=None, instance_info={},
                                 extrinsics=metadata, camera_params=params)
                labels = [('rgb', 'rgb'), ('distance_to_image_plane', 'depth')]
                if include_privileged:
                    labels.append(('instance_segmentation', 'instance_segmentation'))
                for label, key in labels:
                    if label not in anns: continue
                    data = anns[label].get_data()
                    if isinstance(data, dict):
                        if include_privileged:
                            frame['instance_info'] = data.get('info', {})
                        data = data.get('data')
                    if data is None or np.asarray(data).shape[:2] != (h, w):
                        raise RuntimeError(f'Camera {name} has invalid {label} dimensions')
                    frame[key] = np.asarray(data).copy()
                    if key == 'rgb':
                        if frame[key].ndim != 3 or frame[key].shape[2] not in (3, 4):
                            raise RuntimeError(f'Camera {name} has invalid RGB channels')
                        frame[key] = frame[key][..., :3]
                if include_privileged:
                    self.last_metadata[name] = metadata
                frames[name] = frame
            except Exception as exc:
                if not tolerate_errors:
                    raise
                errors[name] = str(exc)
        self.metrics['data_read_wall_s'] += time.perf_counter()-started
        self.metrics['data_reads'] += 1
        return dict(cameras=frames, camera_errors=errors, simulation_time_s=self.rendered_time,
                    physics_frame_index=self.rendered_tick, render_frame_index=self.render_frame_index)

    def capture(self, prefix='settled', refresh=True):
        observations = self.observe(refresh=refresh)
        if not observations['cameras']: return {}
        started = time.perf_counter()
        stats = {}
        base = self.env.output/prefix
        base.parent.mkdir(parents=True, exist_ok=True)
        for name, frame in observations['cameras'].items():
            rgb, depth, seg = frame['rgb'], frame['depth'], frame['instance_segmentation']
            report = dict(rgb_shape=list(rgb.shape) if rgb is not None else None,
                          depth_shape=list(depth.shape) if depth is not None else None,
                          instance_shape=list(seg.shape) if seg is not None else None,
                          intrinsics=frame['intrinsics'], extrinsics=frame['extrinsics'])
            if depth is not None:
                valid = np.isfinite(depth) & (depth > 0)
                if not valid.any(): raise RuntimeError(f'Camera {name} has no finite depth')
                report.update(finite_depth_fraction=float(valid.mean()), min_depth_m=float(depth[valid].min()),
                              max_depth_m=float(depth[valid].max()))
            if self.settings['save_images']:
                from PIL import Image
                if rgb is not None: Image.fromarray(rgb).save(str(base)+f'_{name}_rgb.png')
                if depth is not None:
                    np.save(str(base)+f'_{name}_depth_m.npy', depth)
                    preview = np.where(valid, np.clip((3.-depth)/2.5, 0, 1)*255, 0).astype(np.uint8)
                    Image.fromarray(preview).save(str(base)+f'_{name}_depth_preview.png')
                if seg is not None:
                    np.save(str(base)+f'_{name}_instances.npy', seg)
                    self.env._write_json(f'{prefix}_{name}_instances.json', frame['instance_info'])
            self.env._write_json(f'{prefix}_{name}_calibration.json', frame['extrinsics'])
            self.env._write_json(f'{prefix}_{name}_camera_params.json', frame['camera_params'])
            stats[name] = report
        self.env._write_json(f'{prefix}_sensor_report.json', stats)
        self.metrics['save_wall_s'] += time.perf_counter()-started
        self.metrics['saved_frames'] += 1
        if self.metrics['saved_frames'] == 1: self._gpu_sample('after_first_capture')
        self.env._write_json('camera_rig.json', self.describe())
        self.env._write_json('camera_performance.json', self.performance())
        return stats

    def record_if_due(self):
        if self.sensors and self.settings['recording_enabled'] and self.env.ticks % self.settings['capture_interval_steps'] == 0:
            self.capture(f'recorded/frame_{self.env.ticks:08d}', refresh=False)

    def describe(self):
        return dict(formal_camera_names=list(FORMAL_CAMERAS), resolved_hand_paths=self.hand_paths,
                    hierarchy_driven=True, per_step_world_pose_writes=False, settings=self.settings,
                    cameras={n: {**c['descriptor'], 'last_frame': self.last_metadata.get(n)} for n, c in self.cameras.items()})

    def _gpu_sample(self, label):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                self.metrics['gpu_memory'].append(dict(label=label, used_mib=[int(s) for s in result.stdout.splitlines()]))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass

    def performance(self):
        m = self.metrics.copy()
        for category, count in (('render', 'render_calls'), ('data_read', 'data_reads'), ('save', 'saved_frames')):
            m[category+'_mean_wall_s'] = m[category+'_wall_s']/max(1, m[count])
        m['active_render_products'] = len(self.sensors)
        return m

    def close(self):
        self._closed = True
        self._timeline_subscription = None
        self._release_sensors()

    def _release_sensors(self):
        sensors = list(self.sensors.values())
        self.sensors.clear()
        for sensor in sensors:
            for ann in sensor['annotators'].values(): ann.detach([sensor['product']])
            sensor['product'].destroy()
