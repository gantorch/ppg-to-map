# PPG to MAP Prediction

mean arterial pressure (map) from ppg signals. 

## Structure

```
ppg-to-map/
├── data/
│   ├── raw/              # Original dataset (WFDB format)
│   ├── processed/        # Cleaned/transformed data
│   └── external/         # Third-party datasets
├── scripts/
│   └── download_data.py  # Dataset download
├── src/                  # Source code
├── notebooks/            # Jupyter notebooks
├── ppgtest/              # Virtual environment
├── requirements.txt
└── README.md
```

## Setup

```bash
# Virtual environment
python3 -m venv ppgtest
source ppgtest/bin/activate  # macOS/Linux
# ppgtest\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset

**Source:** [PhysioNet - Pulse Transit Time PPG](https://physionet.org/files/pulse-transit-time-ppg/1.1.0/)

- Format: WFDB (`.dat`, `.hea`, `.atr`)
- 22 subjects (s1-s22)
- 3 activities: run, sit, walk
- Full dataset: ~400 MB

### Download

```bash
python scripts/download_data.py
```

Default: 2 subjects, "sit" only (~13 MB)

To download more, edit `scripts/download_data.py`:

```python
SUBJECTS = [1, 2, 3, 4, 5]  # subjects 1-22
ACTIVITIES = ['sit', 'walk', 'run']
```

## Usage

```python
import wfdb
import pandas as pd

# Load record
record = wfdb.rdrecord('data/raw/s1_sit')

# To DataFrame
df = pd.DataFrame(record.p_signal, columns=record.sig_name)
df['time'] = np.arange(len(df)) / record.fs
```

## Key Dependencies

- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scipy` - Scientific computing
- `wfdb` - WFDB file reading

Full list in `requirements.txt`
