import random
"""Only current rectified grayscale stereo images; no pose/IMU/GT input."""
from pathlib import Path
import os,sys,socket,argparse,json,time,hashlib,signal,traceback,resource
N=Path(__file__).resolve().parents[1];sys.path.insert(0,str(N/'frontend/network'));os.environ['XFORMERS_DISABLED']='1'
import numpy as np,cv2,torch
from omegaconf import OmegaConf
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--output',required=True);a=p.parse_args();o=Path(a.output);o.mkdir(parents=True,exist_ok=True)
torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.enabled=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.set_num_threads(1);cv2.setNumThreads(1);seed=int(os.environ.get("VIO_RESEARCH_SEED","42"));torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);cv2.setRNGSeed(seed);w=N/'weights/23-36-37/model_best_bp2_serialize.pth';model=torch.load(w,map_location='cpu',weights_only=False);model.load_state_dict(model.state_dict(),strict=True);model.args.valid_iters=8;model.args.max_disp=416;model.cuda().eval();count=0;began=time.time()
sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.bind(a.socket);os.chmod(a.socket,0o666);sock.listen(2)
(o/'ready.json').write_text(json.dumps(dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_deterministic=torch.backends.cudnn.deterministic,cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),inference_seed=seed,training_seed=None,strict_state_roundtrip=True,parameters=sum(x.numel() for x in model.parameters()),weight_sha256=hashlib.sha256(w.read_bytes()).hexdigest(),source_pin='476f4249561f7c79ca707326954f9255643412a6',license='NVIDIA research-only',inputs='Current native rectified left/right gray images and native feature coordinates only',GT_input=False,future_frames=False,iterations=8,max_disp=416,temporal_encoder=False,shared_stereo_descriptor_proposals=True,extra_encoder_passes=0,own_training=False),indent=2))
def close(*args):
 (o/'resource.json').write_text(json.dumps(dict(requests=count,wall_s=time.time()-began,max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_peak=torch.cuda.max_memory_allocated())));sock.close();Path(a.socket).unlink(missing_ok=True);sys.exit(0)

# Retain sampled descriptors for actual native feature identities; full features
# only for the current image. Reuses the existing stereo encoder, no extra pass.
from A57_grid_tracking import Grid
extra_grid=Grid(o/'a57_grid')
feature_capture={};frame_cache={};current_feature=None;current_meta=None

def capture(module,inputs,output):
 if inputs[0].shape[0]==2:feature_capture['left']=output[0][:1].detach()
model.feature.register_forward_hook(capture)

def temporal_request(f,head):
 host,target=float(head[1]),float(head[2]);n=int(head[3]);query=np.array([[float(v) for v in f.readline().decode().split()] for _ in range(n)],np.float64).reshape(n,6)
 assert host in frame_cache and target in frame_cache and current_meta['stamp']==target,(host,target,current_meta['stamp'])
 prev=frame_cache[host];cur=frame_cache[target];hi=query[:,0].astype(int);ti=query[:,1].astype(int)
 assert np.all((hi>=0)&(hi<len(prev['q']))) and np.all((ti>=0)&(ti<len(cur['q'])))
 assert np.max(np.abs(prev['q'][hi]-query[:,2:4]))<.003 and np.max(np.abs(cur['q'][ti]-query[:,4:6]))<.003
 started=time.perf_counter();h4,w4=current_feature.shape[-2:];ww,hh,width,height,pads=current_meta['shape'];sx=(ww+pads[0]+pads[1])/w4;sy=(hh+pads[2]+pads[3])/h4
 center=np.column_stack([(query[:,4]*ww/width+pads[0])/sx,(query[:,5]*hh/height+pads[2])/sy]);offset=np.stack(np.meshgrid(np.arange(-8,9),np.arange(-8,9)),axis=-1).reshape(-1,2)
 coords=torch.tensor(center[:,None]+offset[None],device='cuda',dtype=torch.float32);grid=coords.clone();grid[...,0]=grid[...,0]/(w4-1)*2-1;grid[...,1]=grid[...,1]/(h4-1)*2-1
 with torch.inference_mode():
  desc=prev['descriptors'][hi].to(device='cuda',dtype=torch.float32);sample=torch.nn.functional.grid_sample(current_feature.float(),grid.reshape(1,n,-1,2),align_corners=True)[0].permute(1,2,0);scores=torch.nn.functional.cosine_similarity(desc[:,None],sample,dim=-1);valid=(grid.abs()<=1).all(-1);scores[~valid]=-1e4;best=scores.argmax(-1);chosen=coords[torch.arange(n,device='cuda'),best].cpu().numpy();good=valid[torch.arange(n,device='cuda'),best].cpu().numpy()
 alt=np.column_stack([(chosen[:,0]*sx-pads[0])*width/ww,(chosen[:,1]*sy-pads[2])*height/hh]).astype(np.float32).reshape(-1,1,2);src=query[:,2:4].astype(np.float32).reshape(-1,1,2)
 # v25 independent candidate refinement: native feature identity is not replaced.
 alt,ok,_=cv2.calcOpticalFlowPyrLK(prev['image'],cur['image'],src,alt,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01),flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
 back,bok,_=cv2.calcOpticalFlowPyrLK(cur['image'],prev['image'],alt,src.copy(),winSize=(21,21),maxLevel=1,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01),flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
 alt=alt.reshape(-1,2);good&=ok.ravel().astype(bool)&bok.ravel().astype(bool)&np.isfinite(alt).all(1)&(np.linalg.norm(back.reshape(-1,2)-src.reshape(-1,2),axis=1)<=.5)&(alt[:,0]>=1)&(alt[:,0]<width-1)&(alt[:,1]>=1)&(alt[:,1]<height-1)
 alt[~good]=query[~good,4:6]
 sampled=cv2.remap(d,(alt[:,0]*ww/width).reshape(-1,1),(alt[:,1]*hh/height).reshape(-1,1),cv2.INTER_LINEAR).reshape(-1)*width/ww
 seed_right=alt[:,0]-sampled;seed_right[~(np.isfinite(sampled)&(sampled>0)&(seed_right>=0))]=-1
 payload=''.join(f'{x:.9g} {y:.9g} {int(ok)} {sr:.9g}\n' for (x,y),ok,sr in zip(alt,good,seed_right)).encode();f.write((str(n)+'\n').encode());f.write(payload);f.flush()
 with (o/'perf.jsonl').open('a') as log:log.write(json.dumps(dict(kind='temporal',host=host,current_time=target,points=n,valid=int(good.sum()),total_ms=1000*(time.perf_counter()-started)))+'\n')

signal.signal(signal.SIGTERM,close)
while True:

 c,_=sock.accept()
 try:
  f=c.makefile('rwb');head=f.readline().decode().split()
  if head[0]=='T':
   temporal_request(f,head);continue
  stamp=float(head[0]);width,height,n=map(int,head[1:]);q=np.array([[float(v) for v in f.readline().decode().split()] for _ in range(n)],np.float32);left=np.frombuffer(f.read(width*height),np.uint8).reshape(height,width);right=np.frombuffer(f.read(width*height),np.uint8).reshape(height,width);start=time.perf_counter();scale=min(1.,640/width);ww,hh=round(width*scale),round(height*scale)
  ims=[torch.from_numpy(cv2.resize(im,(ww,hh))).cuda()[None,None].expand(1,3,-1,-1).permute(0,2,3,1).contiguous().permute(0,3,1,2).float() for im in [left,right]];pad=InputPadder(ims[0].shape,divis_by=32,force_square=False);ims=pad.pad(*ims)
  t_pre=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):disp=model.forward(*ims,iters=8,test_mode=True,optimize_build_volume='pytorch1')
  t_fwd=time.perf_counter()
  assert 'left' in feature_capture
  current_feature=feature_capture.pop('left');h4,w4=current_feature.shape[-2:];pads=list(pad._pad);sx=(ww+pads[0]+pads[1])/w4;sy=(hh+pads[2]+pads[3])/h4
  grid=torch.tensor(np.column_stack([(q[:,0]*ww/width+pads[0])/sx,(q[:,1]*hh/height+pads[2])/sy]),device='cuda',dtype=torch.float32);grid[:,0]=grid[:,0]/(w4-1)*2-1;grid[:,1]=grid[:,1]/(h4-1)*2-1
  with torch.inference_mode():descriptors=torch.nn.functional.grid_sample(current_feature.float(),grid.reshape(1,n,1,2),align_corners=True)[0,:,:,0].T.detach().half().cpu()
  frame_cache[stamp]=dict(q=q.copy(),descriptors=descriptors,image=left.copy());current_meta=dict(stamp=stamp,shape=(ww,hh,width,height,pads))

  d=pad.unpad(disp.float())[0,0].cpu().numpy();extra_grid.publish(stamp,left,q,d);v=cv2.remap(d,(q[:,0]*ww/width).reshape(-1,1),(q[:,1]*hh/height).reshape(-1,1),cv2.INTER_LINEAR).reshape(-1)*width/ww;result=q.copy();valid=np.isfinite(v)&(v>0)&(q[:,0]-v>=0);result[valid,0]-=v[valid];result[~valid]=-1.
  f.write((str(n)+'\n').encode());payload=''.join(f'{x:.9g} {y:.9g}\n' for x,y in result).encode();f.write(payload);f.flush();count+=1
  t_post=time.perf_counter()
  with (o/'perf.jsonl').open('a') as log:log.write(json.dumps(dict(kind='stereo',time=stamp,points=n,valid=int(valid.sum()),pre_ms=1000*(t_pre-start),fwd_ms=1000*(t_fwd-t_pre),post_ms=1000*(t_post-t_fwd),total_ms=1000*(t_post-start)))+'\n')
 except Exception:
  with (o/'errors.log').open('a') as log:log.write(traceback.format_exc())
  try:c.sendall(b'0\n')
  except OSError:pass
 finally:c.close()
