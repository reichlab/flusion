# This script executes one retrospective run of the main gbq_qr model
# and saves information about feature importances.

# This script should be run with code/gbq as the working directory:
# python retrospective-experiments/gbq_qr_feat_importance.py

import os

ref_date = '2024-01-06'
output_root = '../../retrospective-hub/model-output'
artifact_store_root = '../../retrospective-hub/model-artifacts'

command = f'python gbq.py --ref_date {ref_date} --output_root {output_root} --artifact_store_root {artifact_store_root} --save_feat_importance'

os.system(command)
