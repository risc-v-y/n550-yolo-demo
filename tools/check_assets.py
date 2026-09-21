"""Verify the frozen deployment graph and constants without importing ML packages."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'model/manifest.json').read_text(encoding='utf-8-sig'))
for entry in manifest['files']:
    path=root/'model'/entry['name']
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if path.stat().st_size!=entry['bytes'] or digest!=entry['sha256']:
        raise RuntimeError(f'Frozen model asset changed: {path.name}')
print('Frozen model graph/constants hashes verified')
