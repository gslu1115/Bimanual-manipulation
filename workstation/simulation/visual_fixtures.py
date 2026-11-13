"""Diagnostic INITIAL conditions only. Never consumed by online perception/skills."""
import math
from workstation.task_logic import BoxState


def initial_cartons(config, fixture):
    if fixture == 'clutter':
        from workstation.task_logic import release_states
        return release_states(config,config['seed'])
    entries = [('upright',(-.2,-.06),20)] if fixture == 'upright' else [(fixture,(-.2,-.06),20)]
    if fixture=='upright' and config.get('diagnostic_single_start_side')=='right':
        entries=[('upright',(.20,-.06),20)]
    if fixture == 'upright-pair':
        entries=[('upright',(-.20,-.06),20),('upright',(.20,-.06),-25)]
    elif fixture == 'separated':
        entries=[('upright',(-.2,-.06),20),('side',(0.,-.06),-25),('inverted',(.2,-.06),45)]
    elif fixture == 'separated-upright':
        entries=[('upright',(-.2,-.06),20),('upright',(0.,-.06),-25),('upright',(.2,-.06),45)]
    if config['seed'] != 6:
        import numpy as np
        rng=np.random.default_rng(config['seed'])
        entries=[(posture,tuple(np.asarray(xy)+rng.uniform([-.025,-.04],[.025,.05])),
                  float(rng.uniform(-180.,180.))) for posture,xy,_ in entries]
    if 'diagnostic_visual_yaw_deg' in config:
        if fixture!='upright':raise ValueError('Explicit yaw is only for the upright single-carton fixture')
        yaw=float(config['diagnostic_visual_yaw_deg'])
        if not math.isfinite(yaw):raise ValueError('Visual fixture yaw must be finite')
        entries=[(posture,xy,yaw%360.) for posture,xy,_ in entries]
    result=[]
    variant=config.get('diagnostic_visual_carton_variant')
    spec=next((v for v in config['boxes'].get('variants',[]) if v['name']==variant),None) if variant else None
    if variant and (spec is None or spec['group']!='small' or fixture!='upright'):
        raise ValueError('Single upright fixture requires a configured small-carton variant')
    if fixture=='upright-pair':
        catalogue=[v for v in config['boxes']['variants'] if v['group']=='small']
        if len(catalogue)<2:raise ValueError('The upright pair needs two small variants')
        offset=int(config['seed'])%len(catalogue)
        pair=[catalogue[offset],catalogue[(offset+1)%len(catalogue)]]
        variants=config.get('diagnostic_pair_carton_variants')
        if variants is not None:
            by_name={v['name']:v for v in catalogue}
            if not isinstance(variants,list) or len(variants)!=2 or any(
                    not isinstance(name,str) or name not in by_name for name in variants):
                raise ValueError('Pair fixture requires two configured small-carton variant names')
            pair=[by_name[name] for name in variants]
    for index,(posture,xy,yaw_deg) in enumerate(entries):
        if fixture=='upright-pair':spec=pair[index]
        size=spec['size_m'] if spec else config['boxes']['size']
        roll={'upright':0.,'side':math.pi/2,'inverted':math.pi}[posture]
        yaw=math.radians(yaw_deg)
        height=size[1 if posture == 'side' else 2]
        q=[math.cos(yaw/2)*math.cos(roll/2),math.cos(yaw/2)*math.sin(roll/2),
           math.sin(yaw/2)*math.sin(roll/2),math.sin(yaw/2)*math.cos(roll/2)]
        state=BoxState(f'box_{index:02d}',[*xy,config['table']['top_z']+height/2+.004],
                       q,[0.,0.,0.],[0.,0.,0.])
        if spec:
            from workstation.simulation.cartons import surface_area
            state.size_m=list(size);state.variant=spec['name']
            state.mass_kg=config['boxes']['mass_kg']*surface_area(size)/surface_area(config['boxes']['size'])
        result.append(state)
    return result
