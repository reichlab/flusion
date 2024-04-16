import os
from pathlib import Path
import pandas as pd

def test_create_file(tmp_path):
    os.system(f'python gbq_qr.py --ref_date 2024-03-30 --short_run --output_root {tmp_path}')
    actual_df = pd.read_csv(tmp_path / 'UMass-gbq_qr' / '2024-03-30-UMass-gbq_qr.csv')
    expected_df = pd.read_csv(Path('tests') / 'test_gbq_qr' / '2024-03-30-UMass-gbq_qr.csv')
    assert actual_df.equals(expected_df)
