"""Extract the user-supplied nested dataset ZIP without trusting member paths."""
import argparse
import io
from pathlib import Path, PurePosixPath
import zipfile

def extract(archive, output):
    output = output.resolve()
    with zipfile.ZipFile(archive) as z:
        nested = [n for n in z.namelist() if PurePosixPath(n).name == "dataset.zip"]
        if nested:
            if len(nested) != 1:
                raise ValueError("Ambiguous nested dataset.zip")
            return extract(io.BytesIO(z.read(nested[0])), output)
        for item in z.infolist():
            parts = PurePosixPath(item.filename.replace("\\", "/"))
            destination = (output / str(parts)).resolve()
            if parts.is_absolute() or not destination.is_relative_to(output):
                raise ValueError("Unsafe archive path: " + item.filename)
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(f"Will not overwrite {destination}")
                destination.write_bytes(z.read(item))
    print("Extracted into", output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    extract(args.archive, args.output)
