"""Isaac Sim 6.1 workstation. Import only AFTER SimulationApp is started."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import carb
import omni.usd
import omni.timeline
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux, PhysxSchema
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

from task_logic import BoxState, IndexingConveyor, TaskLoop, norm, release_states, slot_errors


class SortingEnvironment:
    def __init__(self, app, config, output, robot_usd=None, sensors=True, conveyor_test=False, single_pnp=False, dual_handover=False):
        self.app, self.c, self.output = app, config, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.seed = config['seed']
        self.time = 0.
        self.ticks = 0
        self.safe_to_index = True
        self.conveyor = IndexingConveyor(config)
        self.task = TaskLoop()
        self.sensors = {}
        self.materials = {}
        self.timeline = omni.timeline.get_timeline_interface()
        omni.usd.get_context().new_stage()
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(self.stage, '/World')
        self.stage.SetDefaultPrim(root.GetPrim())
        root.GetPrim().SetCustomDataByKey('task', 'dual_panda_clutter_sorting')
        scene = UsdPhysics.Scene.Define(self.stage, '/World/PhysicsScene')
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(config['physics']['gravity'])
        physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx.CreateTimeStepsPerSecondAttr(round(1/config['physics']['dt']))
        physx.CreateEnableCCDAttr(True)
        physx.CreateEnableGPUDynamicsAttr(False)
        physx.CreateBroadphaseTypeAttr('MBP')
        physx.CreateSolverTypeAttr('TGS')
        self.contact = UsdShade.Material.Define(self.stage, '/World/Materials/Contact')
        mat = UsdPhysics.MaterialAPI.Apply(self.contact.GetPrim())
        mat.CreateStaticFrictionAttr(config['physics']['static_friction'])
        mat.CreateDynamicFrictionAttr(config['physics']['dynamic_friction'])
        mat.CreateRestitutionAttr(config['physics']['restitution'])
        self._station()
        self._robots(robot_usd)
        self.initial_states = release_states(config, self.seed)
        if single_pnp:
            p = config['single_pick_place']
            yaw = math.radians(p['box_yaw_deg'])
            self.initial_states = [BoxState('box_00', p['box_xy'] + [config['table']['top_z'] +
                                      config['boxes']['size'][2]/2 + .004],
                                      [math.cos(yaw/2), 0, 0, math.sin(yaw/2)], [0, 0, 0], [0, 0, 0])]
        if dual_handover:
            p = config['dual_handover']
            self.initial_states = [BoxState('box_00', p['box_xy'] + [config['table']['top_z'] +
                                      config['boxes']['size'][2]/2 + .004],
                                      [0, 1, 0, 0], [0, 0, 0], [0, 0, 0])]
        if conveyor_test:
            # Isolated conveyor fixture, explicitly not a robot PnP demonstration.
            self.initial_states = [BoxState(f'box_{i:02d}', xy + [config['conveyor']['top_z'] + config['boxes']['size'][2]/2+.004],
                                           [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])
                                   for i, xy in enumerate(config['conveyor']['slots_xy'])]
        self.box_paths = []
        for state in self.initial_states:
            self._box(state)
        self.boxes = RigidPrim(self.box_paths)
        self._cameras(sensors)
        set_camera_view(eye=[1.75, -2.35, 2.1], target=[0.1, -0.02, .80])
        self.stage.GetRootLayer().Export(str(self.output / 'initial.usda'))
        self._write_json('config_used.json', config)
        self._write_json('release_states.json', [b.to_dict() for b in self.initial_states])

    def _material(self, color, roughness=.7, metallic=0.):
        key = (tuple(color), roughness, metallic)
        if key not in self.materials:
            path = f'/World/Materials/Visual_{len(self.materials):02d}'
            material = UsdShade.Material.Define(self.stage, path)
            shader = UsdShade.Shader.Define(self.stage, path+'/Shader')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            self.materials[key] = material
        return self.materials[key]

    def _cube(self, path, pos, size, color, collision=False, metallic=0.):
        cube = UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        binding.Bind(self._material(color, metallic=metallic))
        if collision:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            api = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            api.CreateContactOffsetAttr(.002)
            api.CreateRestOffsetAttr(0.)
            binding.Bind(self.contact, materialPurpose='physics')
        return cube

    def _station(self):
        c = self.c
        t, belt, u = c['table'], c['conveyor'], c['upstream']
        self._cube('/World/Floor', [0, 0, -.05], [6, 5, .1], [.19, .22, .25], True)
        self._cube('/World/Station/Table', t['center_xy']+[t['top_z']-t['size'][2]/2], t['size'], [.48, .53, .56], True, .45)
        front, back = t['center_xy'][1]-t['size'][1]/2+.065, t['center_xy'][1]+t['size'][1]/2-.065
        for i, (x, y) in enumerate([(-.60, front), (.60, front), (-.60, back), (.60, back)]):
            height = t['top_z']-t['size'][2]
            self._cube(f'/World/Station/Leg_{i}', [x, y, height/2], [.06, .06, height], [.15, .19, .23], True, .5)
        # A shallow unloading tray: an upstream containment feature, not a pose fixture.
        cx, cy = u['center_xy']
        sx, sy = u['tray_size_xy']
        for i, (x, y, dx, dy) in enumerate([(cx-sx/2, cy, .012, sy), (cx+sx/2, cy, .012, sy),
                                           (cx, cy-sy/2, sx, .012), (cx, cy+sy/2, sx, .012)]):
            self._cube(f'/World/Upstream/Rim_{i}', [x, y, t['top_z']+.013], [dx, dy, .026], [.30, .34, .38], True)
        # Conveyor contact surface. Kinematic surface velocity drives contact friction.
        surface = self._cube('/World/Conveyor/Belt', belt['center_xy']+[belt['top_z']-belt['size'][2]/2],
                             belt['size'], [.045, .095, .10], True)
        UsdPhysics.RigidBodyAPI.Apply(surface.GetPrim()).CreateKinematicEnabledAttr(True)
        self.belt_api = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(surface.GetPrim())
        self.belt_api.CreateSurfaceVelocityEnabledAttr(True)
        self.belt_api.CreateSurfaceVelocityLocalSpaceAttr(False)
        self.belt_api.CreateSurfaceVelocityAttr(Gf.Vec3f(0))
        for i, dy in enumerate([-.135, .135]):
            self._cube(f'/World/Conveyor/Frame_{i}', [belt['center_xy'][0], belt['center_xy'][1]+dy, belt['top_z']-.05],
                       [belt['size'][0], .025, .13], [.31, .37, .43], True, .65)
        for i, x in enumerate([-.50, 1.30]):
            self._cube(f'/World/Conveyor/Leg_{i}', [x, belt['center_xy'][1], .33], [.07, .17, .66], [.13, .17, .20], True)
        for i, xy in enumerate(belt['slots_xy']):
            # Coordinate targets only: no renderable geometry, collider, or
            # semantic label. RGB, depth and instance masks cannot see these.
            target = UsdGeom.Xform.Define(self.stage, f'/World/Targets/slot_{i}')
            target.AddTranslateOp().Set(Gf.Vec3d(*xy, belt['top_z']+c['boxes']['size'][2]/2))
            target.GetPrim().CreateAttribute('target:sizeXY', Sdf.ValueTypeNames.Double2).Set(Gf.Vec2d(*belt['slot_size_xy']))
            target.GetPrim().CreateAttribute('target:yawSymmetryDegrees', Sdf.ValueTypeNames.Double).Set(180.)
        # Catch tray at belt exit, including a floor and three sides.
        belt_y = belt['center_xy'][1]
        self._cube('/World/Outfeed/Bottom', [1.65, belt_y, .08], [.5, .55, .08], [.23, .3, .37], True)
        for i, (p, size) in enumerate([([1.90, belt_y, .23], [.025, .55, .30]),
                                      ([1.65, belt_y-.28, .23], [.5, .025, .30]), ([1.65, belt_y+.28, .23], [.5, .025, .30])]):
            self._cube(f'/World/Outfeed/Wall_{i}', p, size, [.23, .3, .37], True)
        UsdLux.DomeLight.Define(self.stage, '/World/Lights/Dome').CreateIntensityAttr(550.)
        light = UsdLux.DistantLight.Define(self.stage, '/World/Lights/Key')
        light.CreateIntensityAttr(1800.)
        light.CreateAngleAttr(12.)
        light.AddRotateXYZOp().Set(Gf.Vec3f(-25, -35, 0))

    def _robots(self, robot_usd):
        if not robot_usd:
            root = get_assets_root_path()
            if not root:
                raise RuntimeError('Isaac asset root unavailable. Supply --robot-usd with an official local Franka USD.')
            robot_usd = root + '/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda'
        self.robot_asset = robot_usd
        self.arms = {}
        for arm in self.c['robots']:
            name, xyz = arm['name'], arm['base_xyz']
            height = xyz[2]-self.c['table']['top_z']
            self._cube(f'/World/Station/{name}_mount', [xyz[0], xyz[1], xyz[2]-height/2], [.18, .18, height], [.13, .17, .22], True)
            path = f'/World/Robots/{name}'
            xf = UsdGeom.Xform.Define(self.stage, path)
            if not xf.GetPrim().GetReferences().AddReference(robot_usd):
                raise RuntimeError(f'Cannot reference Franka asset: {robot_usd}')
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*xyz))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(math.radians(arm['yaw_deg'])/2), 0, 0, math.sin(math.radians(arm['yaw_deg'])/2)))
            articulations = [p for p in Usd.PrimRange(xf.GetPrim()) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
            if not articulations:
                raise RuntimeError(f'Franka reference has no articulation: {robot_usd}')
            self.arms[name] = Articulation(path)
        print(f'[scene] Two Frankas referenced: {robot_usd}', flush=True)

    def _box(self, state):
        path = '/World/Boxes/' + state.name
        self.box_paths.append(path)
        xf = UsdGeom.Xform.Define(self.stage, path)
        xf.AddTranslateOp().Set(Gf.Vec3d(*state.position))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(state.orientation_wxyz[0], Gf.Vec3d(*state.orientation_wxyz[1:])))
        body = UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
        body.CreateVelocityAttr(Gf.Vec3f(*state.linear_velocity))
        UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(self.c['boxes']['mass_kg'])
        physx = PhysxSchema.PhysxRigidBodyAPI.Apply(xf.GetPrim())
        physx.CreateEnableCCDAttr(True)
        # A belt starting below a sleeping body need not wake that body. Keep
        # cartons active so contact surface changes are resolved immediately;
        # stationarity is evaluated by measured velocities, not sleep flags.
        physx.CreateSleepThresholdAttr(0.)
        physx.CreateSolverPositionIterationCountAttr(16)
        physx.CreateSolverVelocityIterationCountAttr(4)
        sx, sy, sz = self.c['boxes']['size']
        self._cube(path+'/Cardboard', [0, 0, 0], [sx, sy, sz], [.56, .34, .16], True)
        # Natural packaging feature; no QR, direction arrow or fiducial.
        self._cube(path+'/TopTape', [0, 0, sz/2+.00010], [sx*.97, .012, .00018], [.69, .49, .27])
        self._cube(path+'/TopSeam', [0, 0, sz/2+.00021], [sx*.97, .0007, .00010], [.25, .16, .08])
        for i, x in enumerate([-sx/2-.0001, sx/2+.0001]):
            self._cube(path+f'/TapeFold_{i}', [x, 0, sz*.30], [.00018, .012, sz*.4], [.69, .49, .27])
        from isaacsim.core.experimental.utils.semantics import add_labels
        add_labels(xf.GetPrim(), labels=['carton'])
        xf.GetPrim().SetCustomDataByKey('box_id', state.name)

    def _cameras(self, enabled):
        if enabled:
            import omni.replicator.core as rep
        for cfg in self.c['cameras']:
            path = '/World/Cameras/'+cfg['name']
            cam = UsdGeom.Camera.Define(self.stage, path)
            matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*cfg['position']), Gf.Vec3d(*cfg['look_at']), Gf.Vec3d(*cfg['up'])).GetInverse()
            cam.AddTransformOp().Set(matrix)
            cam.CreateFocalLengthAttr(cfg['focal_length_mm'])
            cam.CreateHorizontalApertureAttr(cfg['horizontal_aperture_mm'])
            w, h = cfg['resolution']
            cam.CreateVerticalApertureAttr(cfg['horizontal_aperture_mm']*h/w)
            cam.CreateClippingRangeAttr(Gf.Vec2f(.01, 10))
            if enabled:
                product = rep.create.render_product(path, tuple(cfg['resolution']))
                annotators = {}
                for label in ('rgb', 'distance_to_image_plane', 'instance_segmentation', 'camera_params'):
                    ann = rep.AnnotatorRegistry.get_annotator(label)
                    ann.attach([product])
                    annotators[label] = ann
                self.sensors[cfg['name']] = {'product': product, 'annotators': annotators}
            f = cfg['focal_length_mm']/cfg['horizontal_aperture_mm']*w
            # USD row-vector transform -> column-vector OpenCV world transform.
            world_from_usd = np.asarray(matrix).T
            world_from_cv = world_from_usd @ np.diag([1., -1., -1., 1.])
            self._write_json(cfg['name']+'_calibration.json', {
                'resolution_wh': [w, h], 'K': [[f, 0, w/2], [0, f, h/2], [0, 0, 1]],
                'T_world_from_camera_opencv': world_from_cv.tolist(),
                'depth': 'distance_to_image_plane; metres; optical +Z; inf means background',
                'camera_axes': 'OpenCV: right +X, down +Y, forward +Z', 'distortion': [0, 0, 0, 0, 0]})

    def start(self):
        SimulationManager.set_physics_dt(self.c['physics']['dt'])
        SimulationManager.set_device('cpu')
        # Disable automatic stepping: this environment owns its simulation clock.
        self.timeline.set_auto_update(False)
        self.timeline.play()
        self.app.update()
        # A single app update need not have dispatched physics-ready yet.
        views = list(self.arms.values()) + [self.boxes]
        if not all(v.is_physics_tensor_entity_valid() for v in views):
            SimulationManager.initialize_physics()
            for _ in range(20):
                if all(v.is_physics_tensor_entity_valid() for v in views): break
                self.app.update()
        if not all(v.is_physics_tensor_entity_valid() for v in views):
            raise RuntimeError('Physics initialization did not produce valid robot and carton tensor views')
        for name, arm in self.arms.items():
            dofs = arm.get_dof_positions().numpy()
            if dofs.shape[1] != 9:
                raise RuntimeError(f'{name} has {dofs.shape[1]} DOFs; expected 7 arm + 2 fingers')
            arm.set_dof_positions(self.c['robot_home'])
            arm.set_dof_position_targets(self.c['robot_home'])
        self.boxes.set_velocities(linear_velocities=[s.linear_velocity for s in self.initial_states],
                                  angular_velocities=[[0., 0., 0.]]*len(self.initial_states))

    def observe_ground_truth(self):
        p, q = self.boxes.get_world_poses()
        v, w = self.boxes.get_velocities()
        return [BoxState(path.rsplit('/', 1)[1], pi.tolist(), qi.tolist(), vi.tolist(), wi.tolist())
                for path, pi, qi, vi, wi in zip(self.boxes.paths, p.numpy(), q.numpy(), v.numpy(), w.numpy())]

    def step(self, monitor_conveyor=True, render=True):
        dt = self.c['physics']['dt']
        velocity = self.conveyor.update(self.observe_ground_truth(), dt, self.safe_to_index) if monitor_conveyor else 0.
        self.belt_api.GetSurfaceVelocityAttr().Set(Gf.Vec3f(velocity, 0, 0))
        SimulationManager.step()
        self.time += dt
        self.ticks += 1
        if render and self.ticks % 4 == 0:
            RenderingManager.render()
            self.app.update()
        elif self.ticks % 120 == 0:
            self.app.update()

    def settle(self):
        s = self.c['settle']
        stable = 0.
        elapsed = 0.
        dt = self.c['physics']['dt']
        while elapsed < s['timeout_s']:
            self.step(monitor_conveyor=False)
            elapsed += dt
            boxes = self.observe_ground_truth()
            still = all(norm(b.linear_velocity) < s['linear_speed'] and norm(b.angular_velocity) < s['angular_speed'] for b in boxes)
            stable = stable+dt if still and elapsed >= s['min_s'] else 0.
            if stable >= s['hold_s']: break
        boxes = self.observe_ground_truth()
        inside = all(abs(b.position[j]-self.c['upstream']['center_xy'][j]) < self.c['upstream']['tray_size_xy'][j]/2
                     for b in boxes for j in range(2))
        supported = all(b.position[2] > self.c['table']['top_z'] for b in boxes)
        self.settle_report = {'settled': stable >= s['hold_s'], 'elapsed_s': elapsed, 'inside_upstream_tray': inside,
                              'above_table': supported, 'max_linear_speed': max(norm(b.linear_velocity) for b in boxes),
                              'max_angular_speed': max(norm(b.angular_velocity) for b in boxes)}
        self._write_json('settle_report.json', self.settle_report)
        self._write_json('settled_states.json', [b.to_dict() for b in boxes])
        self.export_snapshot('settled.usda')
        print('[settle] '+json.dumps(self.settle_report), flush=True)
        return self.settle_report

    def export_snapshot(self, filename):
        boxes = self.observe_ground_truth()
        # Author actual measured transforms into an exported COPY; never modify live box poses.
        layer = Sdf.Layer.CreateAnonymous()
        layer.TransferContent(self.stage.GetRootLayer())
        snapshot = Usd.Stage.Open(layer)
        for box in boxes:
            xf = UsdGeom.Xformable(snapshot.GetPrimAtPath('/World/Boxes/'+box.name))
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*box.position))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(box.orientation_wxyz[0], Gf.Vec3d(*box.orientation_wxyz[1:])))
            body = UsdPhysics.RigidBodyAPI(xf.GetPrim())
            body.CreateVelocityAttr(Gf.Vec3f(0))
            body.CreateAngularVelocityAttr(Gf.Vec3f(0))
        snapshot.GetRootLayer().Export(str(self.output/filename))

    def capture(self, prefix='settled'):
        if not self.sensors: return {}
        from PIL import Image
        # Render several sensor frames at an unchanged physics state.
        for _ in range(12):
            RenderingManager.render()
            self.app.update()
        stats = {}
        for name, sensor in self.sensors.items():
            anns = sensor['annotators']
            rgb = anns['rgb'].get_data()
            depth = anns['distance_to_image_plane'].get_data()
            if rgb is None or np.asarray(rgb).size == 0 or depth is None or np.asarray(depth).size == 0:
                raise RuntimeError(f'Camera {name} produced no RGB-D data')
            rgb, depth = np.asarray(rgb), np.asarray(depth)
            cfg = next(c for c in self.c['cameras'] if c['name'] == name)
            w, h = cfg['resolution']
            if rgb.shape[:2] != (h, w) or depth.shape[:2] != (h, w):
                raise RuntimeError(f'Invalid sensor dimensions: {rgb.shape}, {depth.shape}')
            Image.fromarray(rgb[..., :3]).save(self.output/f'{prefix}_{name}_rgb.png')
            np.save(self.output/f'{prefix}_{name}_depth_m.npy', depth)
            seg = anns['instance_segmentation'].get_data()
            if isinstance(seg, dict):
                np.save(self.output/f'{prefix}_{name}_instances.npy', seg['data'])
                self._write_json(f'{prefix}_{name}_instances.json', seg['info'])
            valid = np.isfinite(depth) & (depth > 0)
            preview = np.where(valid, np.clip((3.0-depth)/2.5, 0, 1)*255, 0).astype(np.uint8)
            Image.fromarray(preview).save(self.output/f'{prefix}_{name}_depth_preview.png')
            if not valid.any(): raise RuntimeError(f'Camera {name} has no finite depth')
            stats[name] = {'rgb_shape': list(rgb.shape), 'depth_shape': list(depth.shape),
                           'finite_depth_fraction': float(valid.mean()),
                           'min_depth_m': float(depth[valid].min()), 'max_depth_m': float(depth[valid].max())}
        self._write_json(f'{prefix}_sensor_report.json', stats)
        return stats

    def command_joints(self, arm_name, joint_targets):
        """Physical joint-drive command for an external controller; no pose teleport."""
        if self.conveyor.state != 'WAITING':
            raise RuntimeError('Arm motion inhibited during conveyor indexing/fault')
        targets = np.asarray(joint_targets, dtype=float)
        if targets.shape != (9,) or not np.isfinite(targets).all():
            raise ValueError('Expected 9 finite joint position targets')
        self.safe_to_index = False
        self.arms[arm_name].set_dof_position_targets(targets)

    def release_conveyor_interlock(self):
        """Caller must first verify both arms are clear of the conveyor."""
        self.safe_to_index = True

    def report(self):
        boxes = self.observe_ground_truth()
        return {'simulation_time_s': self.time, 'seed': self.seed, 'robot_asset': self.robot_asset,
                'robots': {name: arm.get_dof_positions().numpy().tolist() for name, arm in self.arms.items()},
                'boxes': [b.to_dict() for b in boxes],
                'conveyor': {'state': self.conveyor.state, 'occupied': self.conveyor.occupied,
                             'completed_batches': self.conveyor.completed_batches, 'delivered': sorted(self.conveyor.delivered),
                             'events': self.conveyor.events},
                'evaluation_source': 'simulator_ground_truth',
                'planner': getattr(self, 'planner_name', 'not_connected'), 'phase': self.task.phase,
                'slot_errors': {b.name: [slot_errors(b, xy, self.c) for xy in self.c['conveyor']['slots_xy']] for b in boxes}}

    def _write_json(self, name, data):
        def default(value):
            if hasattr(value, 'tolist'): return value.tolist()
            return str(value)
        (self.output/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default), encoding='utf-8')

    def close(self):
        self.belt_api.GetSurfaceVelocityAttr().Set(Gf.Vec3f(0))
        self.timeline.stop()
