import json
import unittest
import numpy as np
from workstation.app import PROJECT_ROOT
from workstation.simulation.station_priors import depth_backboards
from workstation.planning.collision import station_obstacles,sphere_box_clearance


class StationPriorTests(unittest.TestCase):
    def test_default_environment_has_no_diagnostic_board(self):
        self.assertEqual(depth_backboards({}),())

    def test_diagonal_upper_arm_ray_intersects_actual_board(self):
        board=depth_backboards({'diagnostic_depth_backboard':True})[0]
        origin=np.array([.1132,.2642,1.3446])
        point=np.array([-.2381,-.0654,1.3570])
        direction=point-origin
        near_x=board.centre[0]+board.size[0]/2
        distance=(near_x-origin[0])/direction[0]
        hit=origin+distance*direction
        self.assertGreater(distance,1.)
        self.assertLessEqual(abs(hit[1]-board.centre[1]),board.size[1]/2)
        self.assertLessEqual(abs(hit[2]-board.centre[2]),board.size[2]/2)

    def test_transfer_diagonal_ray_hits_south_return_surface(self):
        board=next(b for b in depth_backboards({'diagnostic_depth_backboard':True}) if b.name.endswith('south'))
        origin=np.array([.1132,.2642,1.3446]);point=np.array([-.1262,-.0840,1.2190])
        direction=point-origin;near_y=board.centre[1]+board.size[1]/2
        distance=(near_y-origin[1])/direction[1]
        hit=origin+distance*direction
        self.assertGreater(distance,1.)
        self.assertLessEqual(abs(hit[0]-board.centre[0]),board.size[0]/2)
        self.assertLessEqual(abs(hit[2]-board.centre[2]),board.size[2]/2)

    def test_same_fixed_board_used_in_render_and_collision_priors(self):
        config=json.loads((PROJECT_ROOT/'config/scene.json').read_text(encoding='utf-8'))
        config['diagnostic_depth_backboard']=True
        boards=depth_backboards(config);self.assertEqual(len(boards),3)
        for board in boards:
            collision=next(b for b in station_obstacles(config) if b.name==board.name)
            self.assertEqual(collision.centre,board.centre);self.assertEqual(collision.size,board.size)
            self.assertGreater(sphere_box_clearance((0.,0,1.1),.10,collision),.9)
            self.assertLess(sphere_box_clearance(board.centre,.01,collision),0)
