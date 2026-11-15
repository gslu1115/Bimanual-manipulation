"""Sampled sphere geometry. Approximate checks are not general collision planning."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AABB:
    name: str
    centre: tuple
    size: tuple
    kind: str = 'fixed'


def partial_carton_extent(obj,scene):
    """Collision bound only: a clipped area need not erase measured axis spans.

    Require fresh masks, near-complete measured spans in all three dimensions
    and consistent horizontal depth. This does not promote UNKNOWN semantics
    to an independently observed pose or provide a grasp/success proof.
    """
    if (set(obj.failures)!={'PARTIAL_OR_MERGED_GEOMETRY'} or
            getattr(obj,'visibility','')!='PARTIAL_OR_UNCERTAIN'):
        return None
    q=obj.quality
    required=('geometric_length_m','geometric_width_m','observed_height_m',
              'observed_length_m','observed_width_m','observed_vertical_span_m',
              'surface_tilt_deg','plane_residual_m','geometry_inlier_ratio','footprint_coverage')
    if not all(k in q and np.isfinite(q[k]) for k in required):return None
    if (getattr(scene,'frame',None)!='workcell' or
            not 0<=scene.assembled_time_s-scene.observation_time_s<=.15 or
            abs(obj.time_s-scene.observation_time_s)>1e-6 or not obj.mask_refs or
            not all(ref in scene.masks and scene.camera_status.get(ref.split('/')[0])=='OK'
                    for ref in obj.mask_refs)):
        return None
    size=np.array([q[k] for k in required[:3]])
    observed=np.array([q[k] for k in required[3:6]])
    if (np.any(size<=0) or not np.allclose(np.sort(size),np.sort(obj.size_m),atol=1e-6,rtol=0) or
            np.max(np.abs(observed[:2]-size[:2]))>.004 or
            abs(observed[2]-size[2])>.008 or not 0<=q['surface_tilt_deg']<=3 or
            not 0<=q['plane_residual_m']<=.002 or not .98<=q['geometry_inlier_ratio']<=1 or
            not .5<=q['footprint_coverage']<=1 or not .003<=obj.uncertainty_m<=.008):
        return None
    # Include the measured span excess and a tilt allowance on every axis.
    return np.maximum(size,observed)+np.linalg.norm(size)*np.sin(np.deg2rad(q['surface_tilt_deg']))


def carton_contact_allowed(a,b):
    """Only carton/carton contact is permitted; unknown kinds remain obstacles."""
    return a.kind=='carton' and b.kind=='carton'


def sphere_box_clearance(centre,radius,box):
    outside = np.maximum(np.abs(np.asarray(centre)-box.centre)-np.asarray(box.size)/2,0.)
    return float(np.linalg.norm(outside)-radius)


def station_obstacles(config):
    # Calibrated fixed scene priors; do not query USD/PhysX collision objects.
    boxes = []
    t,b,u = config['table'],config['conveyor'],config['upstream']
    def add(name,pos,size): boxes.append(AABB(name,tuple(pos),tuple(size)))
    add('floor',[0,0,-.05],[6,5,.1])
    from workstation.simulation.station_priors import depth_backboards
    for board in depth_backboards(config):add(board.name,board.centre,board.size)
    add('table',t['center_xy']+[t['top_z']-t['size'][2]/2],t['size'])
    front,back = t['center_xy'][1]-t['size'][1]/2+.065,t['center_xy'][1]+t['size'][1]/2-.065
    for i,(x,y) in enumerate([(-.6,front),(.6,front),(-.6,back),(.6,back)]):
        height=t['top_z']-t['size'][2]; add(f'table_leg_{i}',[x,y,height/2],[.06,.06,height])
    cx,cy=u['center_xy']; sx,sy=u['tray_size_xy']
    for i,(x,y,dx,dy) in enumerate([(cx-sx/2,cy,.012,sy),(cx+sx/2,cy,.012,sy),
                                   (cx,cy-sy/2,sx,.012),(cx,cy+sy/2,sx,.012)]):
        add(f'tray_rim_{i}',[x,y,t['top_z']+.013],[dx,dy,.026])
    add('belt',b['center_xy']+[b['top_z']-b['size'][2]/2],b['size'])
    for i,dy in enumerate([-.135,.135]):
        add(f'belt_frame_{i}',[b['center_xy'][0],b['center_xy'][1]+dy,b['top_z']-.05],[b['size'][0],.025,.13])
    for i,x in enumerate([-.5,1.3]): add(f'belt_leg_{i}',[x,b['center_xy'][1],.33],[.07,.17,.66])
    by=b['center_xy'][1]
    add('outfeed_bottom',[1.65,by,.08],[.5,.55,.08])
    for i,(pos,size) in enumerate([([1.90,by,.23],[.025,.55,.30]),
            ([1.65,by-.28,.23],[.5,.025,.30]),([1.65,by+.28,.23],[.5,.025,.30])]):
        add(f'outfeed_wall_{i}',pos,size)
    for robot in config['robots']:
        x,y,z=robot['base_xyz']; h=z-t['top_z']
        add(robot['name']+'_mount',[x,y,z-h/2],[.18,.18,h])
    return tuple(boxes)


def scene_obstacles(scene,target_id,extra_margin=.004):
    # Circumscribing yaw-rotated boxes avoids relying on unseen exact surfaces.
    boxes=[]
    for obj in scene.objects:
        if obj.track_id == target_id: continue
        from workstation.perception.scene_estimate import current_cuboid_measurement
        measured=current_cuboid_measurement(obj,scene)
        if measured is not None:
            extent=np.abs(np.asarray(measured['rotation']))@np.asarray(measured['size_m'])
            margin=measured['uncertainty_m']+extra_margin
            boxes.append(AABB(obj.track_id,tuple(measured['position_m']),tuple(extent+2*margin),'carton'))
            continue
        sx,sy,sz=(obj.quality.get('geometric_length_m',obj.size_m[0]),
                  obj.quality.get('geometric_width_m',obj.size_m[1]),
                  obj.quality.get('observed_height_m',obj.size_m[2]))
        if obj.failures or getattr(obj,'visibility','')=='OBSERVED_CUBOID':
            bounded=partial_carton_extent(obj,scene)
            if bounded is None:
                sx=sy=sz=max(obj.size_m)
            else:
                sx,sy,sz=bounded
        c,s=abs(np.cos(obj.axis_yaw_rad)),abs(np.sin(obj.axis_yaw_rad))
        margin=obj.uncertainty_m+extra_margin
        boxes.append(AABB(obj.track_id,obj.position_m,
                          (c*sx+s*sy+2*margin,s*sx+c*sy+2*margin,sz+2*margin),'carton'))
    return boxes


def check_spheres(spheres,other_spheres,obstacles,own_mount,margin=.002,*,rejection=None):
    """(link_index, centre, radius). Connected link groups excluded from self checks."""
    if not spheres: return None,float('inf')
    links=np.asarray([s[0] for s in spheres]);centres=np.asarray([s[1] for s in spheres],float)
    radii=np.asarray([s[2] for s in spheres],float);n=len(spheres)
    static=np.full((n,len(obstacles)),np.inf)
    if obstacles:
        outside=np.maximum(np.abs(centres[:,None,:]-np.asarray([b.centre for b in obstacles]))-
                           np.asarray([b.size for b in obstacles])/2,0.)
        static=np.linalg.norm(outside,axis=2)-radii[:,None]
        # Intended mounting contact only; no hand/forearm body exemption.
        static[(links[:,None] == 0) & np.asarray([b.name == own_mount for b in obstacles])[None,:]]=np.inf
    mutual=np.full((n,len(other_spheres)),np.inf)
    if other_spheres:
        other=np.asarray([s[1] for s in other_spheres],float);other_r=np.asarray([s[2] for s in other_spheres])
        mutual=np.linalg.norm(centres[:,None,:]-other[None,:,:],axis=2)-radii[:,None]-other_r[None,:]
    minimum=min(float(static.min()) if static.size else float('inf'),
                float(mutual.min()) if mutual.size else float('inf'))
    # Keep the original rejection order by sphere, then static before other arm.
    rejected=np.flatnonzero((static < margin).any(axis=1) | (mutual < margin).any(axis=1))
    if len(rejected):
        row=int(rejected[0]);bad=np.flatnonzero(static[row] < margin)
        if rejection is not None and len(bad):
            col=int(bad[0])
            rejection.update(sphere_index=row,link_index=int(links[row]),obstacle=obstacles[col].name,
                             obstacle_kind=obstacles[col].kind,clearance_m=float(static[row,col]),required_m=margin)
        if rejection is not None and not len(bad):
            col=int(np.flatnonzero(mutual[row]<margin)[0])
            rejection.update(sphere_indices=[row,col],link_indices=[int(links[row]),int(other_spheres[col][0])],
                             clearance_m=float(mutual[row,col]),required_m=margin)
        return ('STATIC_PATH_BLOCKED:'+obstacles[int(bad[0])].name if len(bad) else 'OTHER_ARM_COLLISION'),minimum
    distance=np.linalg.norm(centres[:,None,:]-centres[None,:,:],axis=2)-radii[:,None]-radii[None,:]
    eligible=np.triu(np.abs(links[:,None]-links[None,:]) > 2,k=1)
    if eligible.any():
        minimum=min(minimum,float(distance[eligible].min()))
        blocked=eligible & (distance<margin)
        if blocked.any():
            if rejection is not None:
                row,col=map(int,np.argwhere(blocked)[0])
                rejection.update(sphere_indices=[row,col],link_indices=[int(links[row]),int(links[col])],
                                 clearance_m=float(distance[row,col]),required_m=margin)
            return 'SELF_COLLISION',minimum
    return None,minimum
