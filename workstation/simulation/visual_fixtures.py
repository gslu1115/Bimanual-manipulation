"""Diagnostic INITIAL conditions only. Never consumed by online perception/skills."""
import math
from workstation.task_logic import BoxState


def initial_cartons(config, fixture):
    if fixture == 'clutter':
        from workstation.task_logic import release_states
        return release_states(config,config['seed'])
    entries = [('upright',(-.2,-.06),20)] if fixture == 'upright' else [(fixture,(-.2,-.06),20)]
    if fixture == 'separated':
        entries=[('upright',(-.2,-.06),20),('side',(0.,-.06),-25),('inverted',(.2,-.06),45)]
    elif fixture == 'separated-upright':
        entries=[('upright',(-.2,-.06),20),('upright',(0.,-.06),-25),('upright',(.2,-.06),45)]
    if config['seed'] != 6:
        import numpy as np
        rng=np.random.default_rng(config['seed'])
        entries=[(posture,tuple(np.asarray(xy)+rng.uniform([-.025,-.04],[.025,.05])),
                  float(rng.uniform(-180.,180.))) for posture,xy,_ in entries]
    result=[]
    for index,(posture,xy,yaw_deg) in enumerate(entries):
        roll={'upright':0.,'side':math.pi/2,'inverted':math.pi}[posture]
        yaw=math.radians(yaw_deg)
        height=config['boxes']['size'][1 if posture == 'side' else 2]
        q=[math.cos(yaw/2)*math.cos(roll/2),math.cos(yaw/2)*math.sin(roll/2),
           math.sin(yaw/2)*math.sin(roll/2),math.sin(yaw/2)*math.cos(roll/2)]
        result.append(BoxState(f'box_{index:02d}',[*xy,config['table']['top_z']+height/2+.004],
                               q,[0.,0.,0.],[0.,0.,0.]))
    return result
