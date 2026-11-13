"""Sparse, persistent image tracks with learned stereo disparity in rectified pixel units."""
from pathlib import Path
import cv2,numpy as np

class Grid:
    def __init__(self,output):
        self.path=Path(output);self.path.mkdir(exist_ok=True)
        self.image=None;self.pts=np.empty((0,2),np.float32);self.ids=np.empty(0,np.int64);self.serial=0

    def publish(self,t,img,q,d):
        h,w=img.shape;mask=np.full((h,w),255,np.uint8)
        for x,y in np.rint(q).astype(int):cv2.circle(mask,(x,y),5,0,-1)
        pts=self.pts;ids=self.ids
        if len(pts):
            forward,ok,_=cv2.calcOpticalFlowPyrLK(self.image,img,pts,None)
            back,rev,_=cv2.calcOpticalFlowPyrLK(img,self.image,forward,None)
            valid=(ok.ravel()>0)&(rev.ravel()>0)&(np.linalg.norm(back-pts,axis=1)<1)&np.isfinite(forward).all(axis=1)
            pts,ids=forward[valid],ids[valid]
        chosen=[];names=[];cells=set()
        def accept(point,index):
            x,y=np.rint(point).astype(int)
            if x<5 or y<5 or x>=w-5 or y>=h-5 or mask[y,x]==0 or (x//32,y//32) in cells:return
            chosen.append(point);names.append(index);cells.add((x//32,y//32));cv2.circle(mask,(x,y),9,0,-1)
        for pt,i in zip(pts,ids):accept(pt,i)
        if len(chosen)<128:
            corners=cv2.goodFeaturesToTrack(img,256,0.01,10,mask=mask)
            if corners is not None:
                for pt in corners[:,0]:
                    if len(chosen)>=128:break
                    self.serial+=1;accept(pt,self.serial)
        self.pts=np.asarray(chosen,dtype=np.float32).reshape(-1,2);self.ids=np.asarray(names);self.image=img.copy()
        path=self.path/(str(round(t*1e6))+'.txt');tmp=path.with_suffix('.tmp')
        with tmp.open('w') as f:
            if len(chosen):
                xy=self.pts;dh,dw=d.shape
                disparity=cv2.remap(d,(xy[:,0]*dw/w).reshape(-1,1),(xy[:,1]*dh/h).reshape(-1,1),cv2.INTER_LINEAR).ravel()*w/dw
                for i,(x,y),v in zip(names,xy,disparity):
                    f.write(f'{i} {x:.7f} {y:.7f} -1\n')
        tmp.replace(path)
