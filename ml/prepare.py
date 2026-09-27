"""Production preflight: never train or read labels during service startup."""
import json
from pathlib import Path


def main():
    import hashlib
    from ml.model import Predictor
    metadata=json.loads(Path('models/metadata.json').read_text(encoding='utf-8'))
    for member in metadata.get('members', []):
        filename=member['file']
        if Path(filename).name!=filename:
            raise ValueError('Model filename must be a basename')
        digest=hashlib.sha256((Path('models')/filename).read_bytes()).hexdigest()
        if digest!=member['sha256']:
            raise ValueError('Model hash mismatch: '+filename)
    predictor=Predictor()
    print('ML preflight complete: '+predictor.version, flush=True)


if __name__ == "__main__":
    main()
