from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'SHA256SUMS.json').read_text())
for name,expected in manifest.items():
 path=root/name
 assert path.is_file(),name
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
print(f'Verified {len(manifest)} files')
