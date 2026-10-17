import sys, zlib, struct, numpy as np
from modded2model.assets import Assets
from modded2model.schematic import load_structure
from modded2model.voxelize import Voxelizer
def png(path, img):
    h,w,_=img.shape; raw=b''.join(b'\0'+img[y].tobytes() for y in range(h))
    def ch(t,d): return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    open(path,'wb').write(b'\x89PNG\r\n\x1a\n'+ch(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+ch(b'IDAT',zlib.compress(raw))+ch(b'IEND',b''))
def view(g, axis, flip):
    g=np.flip(g,axis) if flip else g
    hit=g.any(axis); d=np.argmax(g,axis).astype(float); n=g.shape[axis]
    shade=1-0.75*d/n
    # edge darkening from depth discontinuities
    gy,gx=np.gradient(d); edge=(np.abs(gx)+np.abs(gy))>0.5
    v=np.where(hit, shade*(1-0.5*edge), 0.15)
    return v
I,P=sys.argv[1],sys.argv[2]
s=load_structure(sys.argv[3]); vx=Voxelizer(Assets([I,P]),8); g=vx.voxelize(s)   # (x,y,z), y up
views=[view(g,1,True).T[::-1], # top: rows z
       view(g,2,False).T[::-1], # front looking +z: (x,y) -> rows y up
       view(g,0,False).T[::-1]] # side looking +x: (y,z)->rows? 
views[2]=view(g,0,False)[::-1] # (y,z): rows y, flip so up is up
H=max(v.shape[0] for v in views); pad=[np.pad(v,((0,H-v.shape[0]),(0,4)),constant_values=0.3) for v in views]
img=np.concatenate(pad,1); sc=max(1,600//img.shape[1]); img=np.kron(img,np.ones((sc,sc)))
rgb=np.stack([img*0.95,img*0.8,img*0.55],-1); png(sys.argv[4],(rgb*255).astype(np.uint8))
