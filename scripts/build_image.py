"""Build the native CPU container and record its local identity."""
from pathlib import Path
import json,subprocess
root=Path(__file__).resolve().parents[1]
tag='bhr-vio:source-release'
subprocess.run(['docker','build','-t',tag,str(root)],check=True)
identity=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',tag],text=True).strip()
(root/'runtime/built_image.json').write_text(json.dumps({'image':tag,'image_id':identity},indent=2))
