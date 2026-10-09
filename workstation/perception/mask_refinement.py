"""Recover narrow visible mask omissions with measured RGB-D evidence only.

No convex hull fill, hidden cuboid projection, renderer labels or unconditional
dilation. The model mask is the seed; every added pixel needs a nearby observed
face plane, similar RGB and a valid metric depth.
"""
import numpy as np


def refine_visible_boundary(frame, mask, xyz, max_pixels=12, max_distance_m=.008, blocked=None):
    import cv2
    mask=np.asarray(mask,bool)
    valid=frame.valid_depth
    du=xyz[1:-1,2:]-xyz[1:-1,:-2];dv=xyz[2:,1:-1]-xyz[:-2,1:-1]
    cross=np.cross(du,dv);length=np.linalg.norm(cross,axis=-1)
    interior=(valid[1:-1,1:-1] & valid[1:-1,2:] & valid[1:-1,:-2] &
              valid[2:,1:-1] & valid[:-2,1:-1] & (length > 1e-12) &
              (np.linalg.norm(du,axis=-1) < .008) & (np.linalg.norm(dv,axis=-1) < .008))
    normals=np.zeros_like(xyz);normal_valid=np.zeros(mask.shape,bool)
    normals[1:-1,1:-1]=cross/np.maximum(length[...,None],1e-12)
    normal_valid[1:-1,1:-1]=interior
    seeds=(cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool) & normal_valid)
    if not seeds.any(): return mask.copy()
    distance,labels=cv2.distanceTransformWithLabels((~seeds).astype(np.uint8),cv2.DIST_L2,5,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
    candidates=(~mask) & valid & (distance <= max_pixels)
    if blocked is not None: candidates &= ~np.asarray(blocked,bool)
    if not candidates.any(): return mask.copy()
    size=int(labels.max())+1
    points=np.zeros((size,3));surface_normals=np.zeros((size,3));colours=np.zeros((size,3))
    points[labels[seeds]]=xyz[seeds]
    surface_normals[labels[seeds]]=normals[seeds]
    colours[labels[seeds]]=frame.rgb[seeds]
    nearest=labels[candidates];delta=xyz[candidates]-points[nearest]
    same_plane=np.abs(np.sum(delta*surface_normals[nearest],axis=1)) <= .001
    close_metric=np.linalg.norm(delta,axis=1) <= max_distance_m
    same_appearance=np.max(np.abs(frame.rgb[candidates].astype(float)-colours[nearest]),axis=1) <= 18
    accepted=same_plane & close_metric & same_appearance
    result=mask.copy();result[candidates]=accepted
    return result
