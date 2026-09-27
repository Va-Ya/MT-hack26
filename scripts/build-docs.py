"""Generate local HTML module docs without network access."""
import os
from pathlib import Path
import pydoc
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
target = root / 'docs/site'
target.mkdir(parents=True, exist_ok=True)
os.chdir(target)
for module in ('ml.features', 'ml.model', 'ml.streaming', 'backend.context', 'backend.scenarios', 'backend.routes', 'backend.ndtp', 'backend.replay'):
    pydoc.writedoc(module)
print(f'Documentation: {target}')
