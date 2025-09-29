import os
import subprocess
from pathlib import Path

# Paths
input_dir = Path(r"doc-old").resolve()
output_dir = Path(r"doc").resolve()

# Make sure output folder exists
output_dir.mkdir(parents=True, exist_ok=True)

print(input_dir)
print(output_dir)

# Iterate through all PDFs in input_dir
for pdf_file in input_dir.glob("*.pdf"):
    output_file = output_dir / pdf_file.name
    print(f"Processing: {pdf_file} -> {output_file}")

    # Build command
    cmd = [
        "ocrmypdf",
        "--force-ocr",  # always OCR even if a text layer exists
        "-l",
        "vie",  # Vietnamese language
        str(pdf_file),
        str(output_file),
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Done: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed on {pdf_file}: {e}")
