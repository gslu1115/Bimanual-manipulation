"""Carton construction data; simulator/evaluation only, never a vision input."""
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class CartonSpec:
    variant: str
    size_m: tuple
    mass_kg: float
    group: str


def surface_area(size):
    x,y,z=size
    return 2*(x*y+x*z+y*z)


def carton_specs(config):
    boxes=config['boxes'];size=tuple(boxes['size']);mass=boxes['mass_kg']
    if boxes.get('size_set','uniform')=='uniform':
        return [CartonSpec('reference',size,mass,'small')]*boxes['count']
    recipe=[]
    for variant in boxes['variants']:
        dims=tuple(variant['size_m'])
        # Same cardboard construction: mass scales with surface area, preserving
        # the original reference mass rather than giving large boxes solid density.
        spec=CartonSpec(variant['name'],dims,mass*surface_area(dims)/surface_area(size),variant['group'])
        recipe.extend([spec]*variant['count'])
    specs=[recipe[i%len(recipe)] for i in range(boxes['count'])]
    random.Random(config['seed']^0xCA470).shuffle(specs)
    return specs


def validate_cartons(boxes):
    def dimensions(size):
        return (isinstance(size,(list,tuple)) and len(size)==3 and
                all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in size))
    if not dimensions(boxes['size']):raise ValueError('Reference carton needs three positive finite dimensions')
    if boxes.get('size_set','uniform') not in ('uniform','mixed'):
        raise ValueError('Carton size_set must be uniform or mixed')
    variants=boxes.get('variants',[])
    if boxes.get('size_set')=='mixed' and not variants:raise ValueError('Mixed cartons require variants')
    names=set()
    for variant in variants:
        name=variant.get('name');size=variant.get('size_m')
        if not isinstance(name,str) or not name or name in names:raise ValueError('Carton variant names must be unique')
        names.add(name)
        if not dimensions(size):raise ValueError('Variant needs three positive finite dimensions')
        if type(variant.get('count')) is not int or variant['count']<1:raise ValueError('Variant count must be positive integer')
        if variant.get('group')=='large':
            if min(size)<=.08:raise ValueError('Large carton must exceed 80 mm on ALL three axes')
        elif variant.get('group')=='small':
            if size[1]>.065 or any(a<b for a,b in zip(size,boxes['size'])):
                raise ValueError('Small carton must be at least reference size with width <=65 mm')
        else:raise ValueError('Carton variant group must be small or large')


def box_size(box,config):
    return box.size_m if box.size_m is not None else config['boxes']['size']


def visual_parts(size):
    """Identical packaging style for every size; only the cardboard collides.

    Tape/seam widths and material colours are the original physical values.
    Tape length and end folds follow each body's dimensions.
    """
    x,y,z=size
    return [
        ('Cardboard',(0.,0.,0.),(x,y,z),(.56,.34,.16),True),
        ('TopTape',(0.,0.,z/2+.00010),(x*.97,.012,.00018),(.69,.49,.27),False),
        ('TopSeam',(0.,0.,z/2+.00021),(x*.97,.0007,.00010),(.25,.16,.08),False),
        ('TapeFold_0',(-x/2-.0001,0.,z*.30),(.00018,.012,z*.4),(.69,.49,.27),False),
        ('TapeFold_1',(x/2+.0001,0.,z*.30),(.00018,.012,z*.4),(.69,.49,.27),False),
    ]


def mixed_release_states(config,seed):
    """Random poses packed into low, non-intersecting release layers.

    Use each rotated body's AABB (with clearance), not the reference size or
    a nine-box tower. These are INITIAL poses; gravity determines the pile.
    """
    from workstation.task_logic import BoxState,rotation
    rng=random.Random(seed);u=config['upstream'];placed=[];states=[]
    for index,spec in enumerate(carton_specs(dict(config,seed=seed))):
        a,v,t=rng.random(),rng.random()*2*math.pi,rng.random()*2*math.pi
        q=[math.sqrt(a)*math.cos(t),math.sqrt(1-a)*math.sin(v),math.sqrt(1-a)*math.cos(v),math.sqrt(a)*math.sin(t)]
        r=rotation(q)
        half=[sum(abs(r[i][j])*spec.size_m[j]/2 for j in range(3)) for i in range(3)]
        limits=[min(u['release_size_xy'][j]/2,u['tray_size_xy'][j]/2-.012-half[j]) for j in range(2)]
        if min(limits)<=0:raise ValueError('Rotated carton cannot fit inside the release tray')
        candidates=[]
        for _ in range(96):
            xy=[u['center_xy'][j]+rng.uniform(-limits[j],limits[j]) for j in range(2)]
            z=config['table']['top_z']+max(u['release_clearance'],half[2]+.04)
            neighbours=[(p,h) for p,h in placed if all(abs(xy[j]-p[j])<half[j]+h[j]+.012 for j in range(2))]
            for p,h in sorted(neighbours,key=lambda item:item[0][2]):
                if abs(z-p[2])<half[2]+h[2]+.012:
                    z=p[2]+max(u['layer_gap'],half[2]+h[2]+.012)
            candidates.append((z,xy))
        z,xy=min(candidates,key=lambda item:item[0])
        position=[*xy,z]
        velocity=[rng.uniform(-u['horizontal_speed'],u['horizontal_speed']) for _ in range(2)]+[0.]
        states.append(BoxState(f'box_{index:02d}',position,q,velocity,[0.,0.,0.],
            size_m=list(spec.size_m),mass_kg=spec.mass_kg,variant=spec.variant))
        placed.append((position,half))
    return states
