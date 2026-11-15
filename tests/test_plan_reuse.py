import unittest
import numpy as np
from workstation.planning.plan_reuse import PreviewPlans,array_key


class PreviewReuseTests(unittest.TestCase):
    def test_changed_start_goal_payload_or_observation_replans_without_using_old_path(self):
        plans=PreviewPlans()
        context=(3,4.,array_key(np.zeros(7)),array_key([.1,.2,.3]),array_key([.01,0,0]))
        original=[np.zeros((2,7)),np.ones((2,7))]
        plans.put('transfer',context,original,{'mode':'parallel'})
        paths,schedule=plans.get('transfer',context)
        self.assertEqual(schedule,{'mode':'parallel'})
        paths[0][:]=9
        self.assertFalse(plans.get('transfer',context)[0][0].any())
        for i in range(len(context)):
            changed=list(context);changed[i]=('changed',)
            self.assertIsNone(plans.get('transfer',tuple(changed)))
        self.assertIsNone(plans.get('lower',context))
