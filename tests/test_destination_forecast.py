import unittest
from types import SimpleNamespace as Box
from unittest.mock import MagicMock,patch
from dataclasses import dataclass
import numpy as np
from workstation.planning.destination_forecast import DestinationForecast


@dataclass
class Candidate:
    candidate_id:str='sample'
    target_id:str='box'


@dataclass
class Spec:
    obj:object
    candidate:object
    lift_q:object
    slot:tuple=(0.,0.)
    def waypoint(self,phase,staging):
        return np.array([self.obj.x,0.,0.]),np.array([1.,0.,0.,0.])


class ForecastGateTests(unittest.TestCase):
    def gate(self,blocked):
        q=np.zeros(7);c=Candidate()
        forecast=DestinationForecast.__new__(DestinationForecast)
        forecast.specs={'a':Spec(Box(x=0.,track_id='box'),c,q)}
        forecast.steps=[('lower',[[q,q]],('a',),('a',),(),None),
                        ('open',[[q,q]],('a',),('a',),(),None)]
        forecast.strategy='orient-first';forecast.counts={'started':1,'passed':1}
        forecast.worker=Box(_cartesian_paths={})
        robot=MagicMock()
        robot.fk.return_value=(np.zeros(3),np.eye(3))
        robot.solve_pose.return_value=q+.01
        robot.joint_knots.side_effect=lambda start,end:[start,end]
        slots=MagicMock(states={(0.,0.):'reserved'})
        report={}
        current={'a':Box(x=.001,track_id='box')}
        with patch('workstation.planning.destination_forecast.check_stage') as validate:
            if blocked:validate.side_effect=RuntimeError('OTHER_ARM_COLLISION')
            forecast.confirm(robot,'fresh_scene',current,slots,{'a':vars(c)},report)
            self.assertEqual(validate.call_count,2)
            for call in validate.call_args_list:
                self.assertEqual(call.args[1],'fresh_scene')
                np.testing.assert_array_equal(call.args[3][0][-1],q+.01)
            self.assertEqual(report['endpoint_corrections'],1)
            self.assertFalse(report['current_frame_replanned'])

    def test_endpoint_connector_and_open_use_fresh_geometry(self):
        self.gate(False)

    def test_blocked_corrected_path_propagates_original_collision(self):
        with self.assertRaisesRegex(RuntimeError,'OTHER_ARM_COLLISION'):
            self.gate(True)
