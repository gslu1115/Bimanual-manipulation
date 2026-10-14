import copy
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tests import test_observed_workspace as fixtures
from tests.test_workspace_short_circuit import exhaustive_reference


class ParallelOccupancyTests(unittest.TestCase):
    def workspace(self):
        ws=fixtures.WorkspaceTests().workspace();original=next(ws._frames())
        ws.packet.cameras={};ws.packet.camera_status={}
        for index in range(3):
            frame=copy.deepcopy(original)
            frame.depth_m=np.full((200,200),1.,np.float32)
            frame.depth_m[80:110,80:110]=.94+index*.01
            frame.valid_depth=np.ones((200,200),bool)
            frame.valid_depth[90:100,90:100]=False
            frame.K=frame.K.copy();frame.K[:2]*=4
            ws.packet.cameras[str(index)]=frame;ws.packet.camera_status[str(index)]='OK'
        return ws

    def test_parallel_camera_union_matches_serial_reference_bit_for_bit(self):
        ws=self.workspace();unknown,occupied=exhaustive_reference(ws)
        with patch('workstation.planning.observed_workspace.ThreadPoolExecutor',wraps=ThreadPoolExecutor) as pool:
            ws.build()
        pool.assert_called_once_with(max_workers=3)
        np.testing.assert_array_equal(ws.unknown,unknown)
        np.testing.assert_array_equal(ws.occupied,occupied)

    def test_worker_failure_never_publishes_a_partial_map(self):
        ws=self.workspace();original=ws._occupied_indices
        bad=ws.packet.cameras['1']
        def checked(frame):
            if frame is bad:raise RuntimeError('injected camera error')
            return original(frame)
        ws._occupied_indices=checked
        with self.assertRaisesRegex(RuntimeError,'injected camera error'):ws._build_occupied()
        self.assertIsNone(ws.occupied)
        ws._occupied_indices=original
        ws._build_occupied();self.assertIsNotNone(ws.occupied)
