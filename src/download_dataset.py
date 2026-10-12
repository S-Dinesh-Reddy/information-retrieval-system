import ssl
import certifi

from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

print("Downloading SciFact dataset...")

context = ssl.create_default_context(cafile=certifi.where())

with urlopen(URL, timeout=60, context=context) as response:    zip_data = response.read()

print("Download complete. Extracting files...")

with ZipFile(BytesIO(zip_data)) as archive:
    archive.extractall(DATA_DIR)

print("Dataset extracted successfully.")
print(f"Location: {DATA_DIR}")

for path in DATA_DIR.rglob("*"):
    if path.is_file():
        print(path.relative_to(DATA_DIR))
