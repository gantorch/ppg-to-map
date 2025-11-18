import os
import urllib.request
import ssl
from pathlib import Path

# Handle SSL certificate issues on macOS
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Base URL
BASE_URL = "https://physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/"

# Download more subjects for better statistical power
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ACTIVITIES = ['sit', 'walk', 'run']

# Get absolute path
script_dir = Path(__file__).parent
project_root = script_dir.parent
data_dir = project_root / "pulse-transit-time-ppg" / "1.1.0" / "csv"
data_dir.mkdir(parents=True, exist_ok=True)

print(f"📂 Downloading CSV files to: {data_dir.absolute()}\n")

def download_file(url, output_path):
    """Download a file from URL to output_path"""
    try:
        print(f"Downloading {output_path.name}...", end=" ", flush=True)
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, output_path)
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ ({file_size:.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

# Download subjects_info.csv first (required)
print("📄 Downloading metadata...")
info_url = BASE_URL + "subjects_info.csv"
info_path = data_dir / "subjects_info.csv"
if not download_file(info_url, info_path):
    print("\n❌ Failed to download subjects_info.csv - required file!")
    exit(1)

print()

# Download CSV files for each subject/activity
downloaded = 0
total = len(SUBJECTS) * len(ACTIVITIES)

for subject in SUBJECTS:
    for activity in ACTIVITIES:
        record_name = f"s{subject}_{activity}"
        print(f"📦 {record_name}:", end=" ")
        
        filename = f"{record_name}.csv"
        url = BASE_URL + filename
        output_path = data_dir / filename
        
        if download_file(url, output_path):
            downloaded += 1

print(f"\n✅ Downloaded {downloaded}/{total} CSV files")
print(f"📂 Files saved to: {data_dir.absolute()}")

