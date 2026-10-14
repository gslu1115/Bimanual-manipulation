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
        with patch('workstation.planning.observed_workspace._occupancy_workers',return_value=8), \
             patch('workstation.planning.observed_workspace.ThreadPoolExecutor',wraps=ThreadPoolExecutor) as pool:
            ws.build()
        self.assertLessEqual(pool.call_args.kwargs['max_workers'],8)
        np.testing.assert_array_equal(ws.unknown,unknown)
        np.testing.assert_array_equal(ws.occupied,occupied)

    def test_worker_failure_never_publishes_a_partial_map(self):
        ws=self.workspace();original=ws._occupied_chunk
        def checked(chunk):
            raise RuntimeError('injected camera error')
        ws._occupied_chunk=checked
        with patch('workstation.planning.observed_workspace._occupancy_workers',return_value=8), \
             self.assertRaisesRegex(RuntimeError,'injected camera error'):ws._build_occupied()
        self.assertIsNone(ws.occupied)
        ws._occupied_chunk=original
        ws._build_occupied();self.assertIsNotNone(ws.occupied)

    def test_uneven_chunks_preserve_edges_invalid_points_and_input_arrays(self):
        ws=self.workspace()
        ws.robot_links=((fixtures.WorkspaceTests().cube(),np.array([0.,0.,.95]),np.eye(3)),)
        xyz=np.random.default_rng(7).uniform(ws.lower-.1,ws.upper+.1,(65539,3))
        xyz[:4]=[ws.lower,ws.upper,ws.upper-1e-8,ws.lower-1e-8]
        valid=np.ones(len(xyz),bool);valid[::11]=False
        xyz[-1]=np.nan;valid[-1]=False
        original=xyz.copy();full=ws._occupied_chunk((xyz,valid))
        chunks=[ws._occupied_chunk((xyz[i:i+32768],valid[i:i+32768])) for i in range(0,len(xyz),32768)]
        np.testing.assert_array_equal(np.concatenate(chunks),full)
        np.testing.assert_array_equal(xyz,original)

    def test_small_input_stays_serial(self):
        ws=fixtures.WorkspaceTests().workspace()
        with patch('workstation.planning.observed_workspace.ThreadPoolExecutor') as pool:
            ws._build_occupied()
        pool.assert_not_called()

    def test_single_cpu_stays_serial_and_matches_parallel(self):
        ws=self.workspace();parallel=self.workspace()
        with patch('workstation.planning.observed_workspace._occupancy_workers',return_value=1), \
             patch('workstation.planning.observed_workspace.ThreadPoolExecutor') as pool:
            ws._build_occupied()
        pool.assert_not_called()
        with patch('workstation.planning.observed_workspace._occupancy_workers',return_value=8):
            parallel._build_occupied()
        np.testing.assert_array_equal(ws.occupied,parallel.occupied)
