"""Shared fixed geometry for explicit acquisition experiments, not carton state."""
from dataclasses import dataclass


@dataclass(frozen=True)
class StationPrimitive:
    name:str
    centre:tuple
    size:tuple
    colour:tuple


def depth_backboards(config):
    """A real rendered/colliding background surface outside the workcell.

    No-return/invalid depth is still unknown. This board supplies a measured
    return in the controlled diagnostic setup; it does not fabricate depth.
    Default environment has no board. Hardware availability is not assumed.
    """
    if not config.get('diagnostic_depth_backboard',False):return ()
    # The staged wrist looks diagonally across the workcell. A 1.2 m board
    # misses valid rays to the upper-arm corridor at y < -0.6 m. This actual
    # 3 m surface is shared with the static collision model, not a depth fill.
    return (StationPrimitive('depth_backboard_west',(-1.15,0.,1.1),(.025,3.,2.2),(.60,.62,.64)),
            StationPrimitive('depth_backboard_east',(1.15,0.,1.1),(.025,3.,2.2),(.60,.62,.64)),
            StationPrimitive('depth_backboard_south',(0.,-1.2,1.1),(3.,.025,2.2),(.60,.62,.64)))
