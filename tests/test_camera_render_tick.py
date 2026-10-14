import importlib.util
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch,MagicMock
import unittest
import numpy as np
from workstation.observations.observation_packet import PANDA_ARM_JOINT_NAMES,PANDA_FINGER_JOINT_NAMES


class CameraRenderTickTests(unittest.TestCase):
    def test_one_backend_update_keeps_image_time_robot_latch(self):
        calls=[]
        app=SimpleNamespace(update=lambda:calls.append('app_update'))
        manager=SimpleNamespace(render=lambda:(calls.append('render'),app.update()))
        modules={name:ModuleType(name) for name in ('omni','omni.timeline','pxr','isaacsim','isaacsim.core','isaacsim.core.rendering_manager')}
        modules['omni'].timeline=modules['omni.timeline']
        modules['isaacsim.core.rendering_manager'].RenderingManager=manager
        for name in ('Gf','Sdf','Usd','UsdGeom','UsdPhysics'):setattr(modules['pxr'],name,MagicMock())
        path=Path(__file__).resolve().parents[1]/'workstation/observations/camera_system.py'
        with patch.dict('sys.modules',modules):
            spec=importlib.util.spec_from_file_location('_camera_render_tick_test',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        states={'arm':SimpleNamespace(qpos=np.arange(9,dtype=float))}
        def observe():calls.append('observe');return states
        sensor=module.CameraSystem.__new__(module.CameraSystem)
        sensor.env=SimpleNamespace(ticks=37,time=.3,observe_robot_states=observe,app=app,
            arms={'arm':SimpleNamespace(dof_names=PANDA_ARM_JOINT_NAMES+PANDA_FINGER_JOINT_NAMES)})
        sensor.metrics={'render_wall_s':0.,'render_calls':0};sensor.render_frame_index=0
        sensor.render()
        self.assertEqual(calls,['observe','render','app_update'])
        self.assertEqual((sensor.rendered_tick,sensor.rendered_time),(37,.3))
        self.assertIs(sensor.rendered_robot_states,states)
        self.assertEqual(sensor.render_frame_index,1)
        np.testing.assert_array_equal(sensor.rendered_joint_positions['arm'],states['arm'].qpos)
