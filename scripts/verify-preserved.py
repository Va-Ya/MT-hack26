"""Check preserved archive files without loading models or modifying any source."""
from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'docs/preserved-sha256.json').read_text(encoding='utf-8'))
failures = []
for name, expected in manifest.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
if failures:
    print('CHANGED OR MISSING: ' + ', '.join(failures))
    sys.exit(1)
print(f'OK: {len(manifest)} original files unchanged (SHA-256).')
