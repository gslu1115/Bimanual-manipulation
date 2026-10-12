"""Controlled side-to-upright attempt using existing full-pose contact phases."""
from workstation.skills.visual_pick_place import VisualPickPlace


class VisualSideAdjustment(VisualPickPlace):
    accepted_postures=('SIDE',)
    skill_name='SIDE_ADJUST'
    run_scope='controlled_side_carton_to_upright_attempt'
    # The rotated closing axis can put a fingertip below the box. This is
    # a proposed goal, still subject to full path and <=20 mm drop-bound gates.
    release_gap_m=.015

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.grasp_source is None or tuple(self.grasp_source.accepted_postures)!=self.accepted_postures:
            raise ValueError('Side adjustment requires the explicit full-pose SIDE GraspGen source')
