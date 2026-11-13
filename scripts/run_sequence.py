"""Run one full stereo-inertial sequence. No GT enters inference."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,uuid
p=argparse.ArgumentParser()
p.add_argument('--dataset',choices=['euroc','kitti'],required=True)
p.add_argument('--sequence',required=True)
p.add_argument('--data',type=Path,required=True,help='mav0 directory with cam0/cam1/imu0')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--gpu',default='0');p.add_argument('--seed',type=int,default=44)
a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.output.resolve()
assert not out.exists(),'Use a new output directory; existing runs are never overwritten.'
assert (root/'runtime/built_image.json').is_file(),'Run scripts/build_image.py first.'
for sensor in ['cam0','cam1','imu0']:assert (a.data/sensor/'data.csv').is_file()
out.mkdir(parents=True);front=out/'learned_frontend';bridge=Path('/tmp/vio_source_map_bridge');bridge.mkdir(exist_ok=True)
socket_name='bhr_'+uuid.uuid4().hex[:12]+'.sock';sock=bridge/socket_name
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,VIO_RESEARCH_SEED=str(a.seed),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED=str(a.seed))
start=time.perf_counter()
with (out/'frontend.log').open('w') as log:
 server=subprocess.Popen([sys.executable,str(root/'runtime/server_packed_perf.py'),'--socket',str(sock),'--output',str(front)],env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  deadline=time.monotonic()+600
  while not ((front/'ready.json').exists() and sock.exists()):
   if server.poll() is not None:raise RuntimeError('Frontend failed. See frontend.log')
   if time.monotonic()>deadline:raise TimeoutError('Frontend startup exceeded 600 s')
   time.sleep(.2)
  subprocess.run([sys.executable,str(root/'runtime/replay_native.py'),'--root',str(root),'--data',str(a.data.resolve()),'--dataset',a.dataset,'--sequence',a.sequence,'--output',str(out/'native'),'--arm','supported_history','--socket-name',socket_name],env=env,check=True)
 finally:
  if server.poll() is None:server.terminate()
  try:server.wait(timeout=30)
  except subprocess.TimeoutExpired:server.kill();server.wait()
  sock.unlink(missing_ok=True)
  (out/'run_time.json').write_text(json.dumps({'wall_s':time.perf_counter()-start,'timing_scope':'diagnostic runner; not the production paper timing harness'},indent=2))
