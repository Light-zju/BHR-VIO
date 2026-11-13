from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'frontend/network'))
import torch
from core.foundation_stereo import FoundationStereo
w=root/'weights/23-36-37/model_best_bp2_serialize.pth'
assert hashlib.sha256(w.read_bytes()).hexdigest()=='af0658f289ec840b292645f8d5538978f06e8cabaa1fd31e84acc91af268e990'
model=torch.load(w,map_location='cpu',weights_only=False)
n=sum(p.numel() for p in model.parameters())
assert n==17654857,n
print(json.dumps({'parameters':n,'torch':torch.__version__,'CUDA_runtime':torch.version.cuda,'checkpoint_verified':True,'inference_run':False},indent=2))
