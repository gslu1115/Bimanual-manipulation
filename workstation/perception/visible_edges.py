"""Measured horizontal/vertical face junctions, distinct from RGB texture edges.

Controlled carton geometry only. Both faces must have valid visible depth within
the carton mask. Hidden edges and a complete cuboid wireframe are not produced.
"""
import numpy as np
from workstation.perception.scene_estimate import ObservedEdge


def observed_face_edges(frame, mask, xyz, mask_ref):
    import cv2
    # Centred metric differences; do not use wrapped/border samples.
    du=xyz[1:-1,2:]-xyz[1:-1,:-2]
    dv=xyz[2:,1:-1]-xyz[:-2,1:-1]
    cross=np.cross(du,dv)
    length=np.linalg.norm(cross,axis=-1)
    vertical_component=np.abs(cross[...,2])/np.maximum(length,1e-12)
    visible=mask & frame.valid_depth
    interior=(visible[1:-1,1:-1] & visible[1:-1,2:] & visible[1:-1,:-2] &
              visible[2:,1:-1] & visible[:-2,1:-1] & (length > 1e-12))
    # Reject depth jumps/near-background derivatives instead of manufacturing a
    # surface normal across the silhouette or an occluding robot finger.
    interior &= (np.linalg.norm(du,axis=-1) < .008) & (np.linalg.norm(dv,axis=-1) < .008)
    horizontal=np.zeros(mask.shape,np.uint8); vertical=np.zeros(mask.shape,np.uint8)
    horizontal[1:-1,1:-1]=interior & (vertical_component > .94)
    vertical[1:-1,1:-1]=interior & (vertical_component < .35)
    junction=horizontal & cv2.dilate(vertical,np.ones((5,5),np.uint8))
    # A one-pixel band absorbs sampling jitter and the small tape thickness.
    # It stays inside measured mask/depth; no long occlusion gap is bridged.
    band=cv2.dilate(junction,np.ones((3,3),np.uint8)) & visible.astype(np.uint8)
    lines=cv2.HoughLinesP(band,rho=1,theta=np.pi/360,threshold=12,
                          minLineLength=18,maxLineGap=4)
    if lines is None: return ()
    result=[]
    ordered=sorted(lines[:,0,:],key=lambda line: -np.linalg.norm(line[2:]-line[:2]))
    for u0,v0,u1,v1 in ordered:
        samples=max(abs(u1-u0),abs(v1-v0))+1
        us=np.rint(np.linspace(u0,u1,samples)).astype(int)
        vs=np.rint(np.linspace(v0,v1,samples)).astype(int)
        support=float(band[vs,us].mean())
        if support < .75: continue
        points=xyz[vs,us]
        if np.linalg.norm(points[-1]-points[0]) < .015: continue
        centre=points.mean(axis=0)
        direction=np.linalg.svd(points-centre,full_matrices=False)[2][0]
        residual=np.linalg.norm((points-centre)-np.outer((points-centre)@direction,direction),axis=1)
        if np.quantile(residual,.9) > .003: continue
        # Suppress shorter overlapping fits to the same metric edge, while
        # retaining separate visible segments on either side of an occlusion.
        duplicate=False
        for edge in result:
            a,b=np.asarray(edge.endpoints_m)
            old_direction=(b-a)/np.linalg.norm(b-a)
            if abs(direction@old_direction) < np.cos(np.deg2rad(5)): continue
            distance=np.linalg.norm((centre-a)-((centre-a)@old_direction)*old_direction)
            if distance > .003: continue
            interval=np.sort((points[[0,-1]]-a)@old_direction)
            overlap=max(0.,min(interval[1],np.linalg.norm(b-a))-max(interval[0],0.))
            if overlap > .7*(interval[1]-interval[0]): duplicate=True; break
        if duplicate:
            continue
        result.append(ObservedEdge(frame.name,mask_ref,'VISIBLE_FACE_INTERSECTION',
            ((int(u0),int(v0)),(int(u1),int(v1))),
            (tuple(float(x) for x in points[0]),tuple(float(x) for x in points[-1])),
            'valid depth normals on both visible carton faces; metric straight-line fit'))
    return _fit_connected_edges(frame,mask,xyz,horizontal,vertical,tuple(result))


def _fit_connected_edges(frame,mask,xyz,horizontal,vertical,edges):
    """One straight intersection per pair of observed planes, split at occlusion.

    Endpoint extension uses visible mask/depth support. It never projects a
    hidden cuboid edge or bridges pixels occupied by a gripper/background.
    """
    import cv2
    groups=[]
    for edge in edges:
        a,b=np.asarray(edge.endpoints_m);direction=(b-a)/np.linalg.norm(b-a)
        group=next((g for g in groups if abs(direction@g[0]) > np.cos(np.deg2rad(5)) and
                    np.linalg.norm((a-g[1])-((a-g[1])@g[0])*g[0]) < .003),None)
        if group is None: groups.append([direction,a,[edge]])
        else: group[2].append(edge)
    h,w=mask.shape;vs,us=np.indices(mask.shape);result=[]
    boundary=mask & ~cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    inv=np.linalg.inv(frame.T_workcell_from_camera_cv)
    def project(point):
        camera=inv[:3,:3]@point+inv[:3,3]
        uv=frame.K@camera
        return uv[:2]/uv[2]
    for _,_,group in groups:
        uv=np.asarray([p for edge in group for p in edge.pixels_uv],float)
        centre=uv.mean(axis=0);direction=np.linalg.svd(uv-centre,full_matrices=False)[2][0]
        along=(us-centre[0])*direction[0]+(vs-centre[1])*direction[1]
        distance=np.abs((us-centre[0])*direction[1]-(vs-centre[1])*direction[0])
        extent=(uv-centre)@direction
        near=(distance < 8) & (along > extent.min()-30) & (along < extent.max()+30)
        planes=[]
        for face in (horizontal,vertical):
            points=xyz[near & face.astype(bool)]
            if len(points) < 30: break
            mean=points.mean(axis=0);normal=np.linalg.svd(points-mean,full_matrices=False)[2][-1]
            residual=np.sqrt(np.mean(((points-mean)@normal)**2))
            if residual > .0018: break
            planes.append((normal,float(normal@mean)))
        if len(planes) != 2:
            result.extend(group);continue
        (n0,d0),(n1,d1)=planes
        line=np.cross(n0,n1);length=np.linalg.norm(line)
        if length < .8: result.extend(group);continue
        line/=length
        reference=np.mean([p for edge in group for p in edge.endpoints_m],axis=0)
        origin=np.linalg.solve(np.stack((n0,n1,line)),np.array([d0,d1,line@reference]))
        uv0=project(origin);delta=project(origin+line*.02)-uv0
        pixel_direction=delta/np.linalg.norm(delta)
        observed=(uv-uv0)@pixel_direction
        steps=np.arange(np.floor(observed.min()-30),np.ceil(observed.max()+30)+1)
        pixels=np.rint(uv0+steps[:,None]*pixel_direction).astype(int)
        inside=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
        supported=np.zeros(len(pixels),bool)
        indices=np.flatnonzero(inside);u,v=pixels[indices].T
        points=xyz[v,u];relative=points-origin
        metric_distance=np.linalg.norm(relative-np.outer(relative@line,line),axis=1)
        supported[indices]=mask[v,u] & frame.valid_depth[v,u] & (metric_distance < .003)
        # Split at every unsupported pixel. Tape sampling noise on a measured
        # plane may pass; a mask hole or an occluding robot cannot pass.
        padded=np.r_[False,supported,False].astype(int);changes=np.diff(padded)
        for first,last in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            if last-first < 18: continue
            endpoints=pixels[[first,last-1]].copy()
            for endpoint,sign in ((0,-1),(1,1)):
                eu,ev=endpoints[endpoint]
                neighbours=boundary & frame.valid_depth & ((us-eu)**2+(vs-ev)**2 <= 36)
                nv,nu=np.nonzero(neighbours)
                if not len(nu): continue
                relative=xyz[nv,nu]-origin
                close=np.linalg.norm(relative-np.outer(relative@line,line),axis=1) < .003
                if not close.any(): continue
                candidates=np.column_stack((nu[close],nv[close]))
                indices=(candidates-uv0)@pixel_direction*sign
                endpoints[endpoint]=candidates[np.argmax(indices)]
            a,b=xyz[endpoints[:,1],endpoints[:,0]]
            if np.linalg.norm(b-a) < .015: continue
            result.append(ObservedEdge(frame.name,group[0].mask_ref,'VISIBLE_FACE_INTERSECTION',
                tuple(tuple(int(v) for v in point) for point in endpoints),
                (tuple(float(v) for v in a),tuple(float(v) for v in b)),
                'two measured planes; connected valid visible depth; supported silhouette endpoints'))
    return tuple(result)
