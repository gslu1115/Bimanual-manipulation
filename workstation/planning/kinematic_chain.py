"""Whole-chain FK from the same URDF and calibrated base used by the solver."""
import xml.etree.ElementTree as ET
import numpy as np


def rpy_rotation(rpy):
    r,p,y=rpy;cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


class KinematicChain:
    def __init__(self,urdf_path,joint_names,frames,base_position,base_rotation,root='base_link'):
        document=ET.parse(urdf_path).getroot()
        joints={j.find('child').get('link'):j for j in document.findall('joint')}
        needed=set()
        for frame in frames:
            current=frame
            while current!=root:
                needed.add(current);current=joints[current].find('parent').get('link')
        self.base_position=np.asarray(base_position,float).copy()
        self.base_rotation=np.asarray(base_rotation,float).copy()
        self.root=root;self.frames=tuple(frames);self.joint_names=tuple(joint_names)
        self.steps=[];done={root};remaining=set(needed)
        while remaining:
            ready=sorted(c for c in remaining if joints[c].find('parent').get('link') in done)
            if not ready:raise ValueError('URDF chain is not rooted at the configured model root')
            for child in ready:
                j=joints[child];origin=j.find('origin');axis=j.find('axis')
                xyz=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
                rpy=np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
                kind=j.get('type');name=j.get('name')
                if kind not in ('fixed','revolute','continuous'):raise ValueError('Unsupported requested chain joint: '+name)
                index=None if kind=='fixed' else self.joint_names.index(name)
                vector=np.fromstring(axis.get('xyz','1 0 0') if axis is not None else '1 0 0',sep=' ')
                vector=np.array([1.,0.,0.]) if index is None else vector/np.linalg.norm(vector)
                x,y,z=vector;skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                self.steps.append((child,j.find('parent').get('link'),xyz,rpy_rotation(rpy),index,skew,skew@skew))
                done.add(child);remaining.remove(child)
        self.skews=np.zeros((len(joint_names),3,3))
        self.squares=np.zeros_like(self.skews)
        self.identity=np.eye(3)
        for _,_,_,_,index,skew,square in self.steps:
            if index is not None:self.skews[index]=skew;self.squares[index]=square

    def poses(self,q):
        q=np.asarray(q,float)
        if q.shape!=(len(self.joint_names),) or not np.isfinite(q).all():raise ValueError('Invalid chain joint state')
        sine,cosine=np.sin(q),np.cos(q)
        rotations=self.identity+sine[:,None,None]*self.skews+(1-cosine[:,None,None])*self.squares
        poses={self.root:(self.base_position,self.base_rotation)}
        for child,parent,xyz,origin,index,skew,square in self.steps:
            p,r=poses[parent]
            rotation=origin if index is None else origin@rotations[index]
            poses[child]=(p+r@xyz,r@rotation)
        result={frame:poses[frame] for frame in self.frames}
        for p,r in result.values():p.flags.writeable=False;r.flags.writeable=False
        return result
