import os
import urllib.request
from pathlib import Path

# Base URL
BASE_URL = "https://physionet.org/files/pulse-transit-time-ppg/1.1.0/"

# Download ONLY a subset for testing (adjust these as needed)
SUBJECTS = [1, 2]  # Just 2 subjects for testing
ACTIVITIES = ['sit']  # Just 1 activity
EXTENSIONS = ['.dat', '.hea', '.atr']

# Get absolute path to ensure files go to the right place
script_dir = Path(__file__).parent
project_root = script_dir.parent
data_dir = project_root / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

print(f"📂 Downloading to: {data_dir.absolute()}\n")

def download_file(url, output_path):
    """Download a file from URL to output_path"""
    try:
        print(f"Downloading {output_path.name}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, output_path)
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ ({file_size:.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

# Download data files
total_files = len(SUBJECTS) * len(ACTIVITIES) * len(EXTENSIONS)
downloaded = 0

print(f"📊 Downloading {len(SUBJECTS)} subjects × {len(ACTIVITIES)} activities = {total_files} files\n")

for subject in SUBJECTS:
    for activity in ACTIVITIES:
        record_name = f"s{subject}_{activity}"
        print(f"📦 {record_name}:")
        
        for ext in EXTENSIONS:
            filename = record_name + ext
            url = BASE_URL + filename
            output_path = data_dir / filename
            
            if download_file(url, output_path):
                downloaded += 1
        print()  # Blank line between records

# Download essential metadata files
print("📄 Downloading metadata...")
for filename in ['README.txt', 'LICENSE.txt']:
    url = BASE_URL + filename
    output_path = data_dir / filename
    download_file(url, output_path)

print(f"\n✅ Successfully downloaded {downloaded}/{total_files} signal files")
print(f"📂 Files saved to: {data_dir.absolute()}")
print(f"\n💡 To download more data, edit SUBJECTS and ACTIVITIES in this script.")