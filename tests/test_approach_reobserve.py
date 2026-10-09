import unittest
from types import SimpleNamespace
from unittest.mock import patch
from workstation.skills.visual_pick_place import VisualPickPlace


class ApproachReobserveTests(unittest.TestCase):
    def test_failed_inspection_preserves_original_motion_rejection(self):
        skill=VisualPickPlace.__new__(VisualPickPlace)
        original=dict(rejection='UNOBSERVED_PATH_SPACE',workspace={'last_unknown_sample_m':[0,0,1]})
        robot=SimpleNamespace(last_plan_check=original,execute_camera_view=lambda *args:None,
                              allow_active_reobserve=True)
        def reject(*args,**kwargs):
            self.assertTrue(kwargs['check_approach_corridor'])
            raise RuntimeError('UNOBSERVED_PATH_SPACE')
        robot.execute_cartesian=reject
        skill.robot=robot;skill.arm='left';skill.held=False
        skill.candidate=SimpleNamespace(target_id='visual_1')
        skill.scene=SimpleNamespace(objects=[SimpleNamespace(track_id='visual_1',failures=[])])
        skill.events=[{'phase':'APPROACH'}];skill.observe=lambda *args:None
        skill.packet=None;skill.priors=None;skill.observer_arms=set()
        def inspection(*args):
            robot.last_plan_check={'rejection':'OTHER_ARM_COLLISION'}
            return skill.scene,dict(success=False,executed=False,reason='NO_VERIFIED_INSPECTION_VIEW')
        with patch('workstation.skills.camera_reobserve.GapReobserve.run',side_effect=inspection):
            with self.assertRaisesRegex(RuntimeError,'UNOBSERVED_PATH_SPACE'):
                skill.move([0,0,1],0,0)
        self.assertEqual(robot.last_plan_check,original)
        self.assertEqual(skill.events[0]['path_reobservation'][0]['original_approach_rejection'],original)

    def test_fixed_camera_default_never_proposes_inspection_motion(self):
        skill=VisualPickPlace.__new__(VisualPickPlace)
        def reject(*args,**kwargs):
            self.assertTrue(kwargs['check_approach_corridor'])
            raise RuntimeError('UNOBSERVED_PATH_SPACE')
        skill.robot=SimpleNamespace(execute_cartesian=reject,execute_camera_view=lambda *args:None)
        skill.arm='left';skill.held=False
        skill.candidate=SimpleNamespace(target_id='visual_1')
        skill.scene=SimpleNamespace(objects=[]);skill.events=[{'phase':'APPROACH'}]
        with patch('workstation.skills.camera_reobserve.GapReobserve.run') as inspection:
            with self.assertRaisesRegex(RuntimeError,'UNOBSERVED_PATH_SPACE'):
                skill.move([0,0,1],0,0)
        inspection.assert_not_called()
