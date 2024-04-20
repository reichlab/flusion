# Predictions for the gbq_qr_no_level model were not generated in real time
# until the reference date 2023-12-02.  This script fills in the missing dates
# to enable a complete evaluation of component model importance.  Retrospective
# model fits are generated using the data that would have been available in
# real time.
#
# To maintain transparency about which model outputs were and were not generated in
# real time, these model outputs are stored in flusion/retrospective-hub.

# This script should be run with code/gbq as the working directory:
# python retrospective-experiments/fill_gbq_no_level.py

import os
import datetime
from multiprocessing import Pool


def run_command(command):
    """Run system command"""
    os.system(command)


missing_ref_dates = [
    (datetime.date(2023, 10, 14) + datetime.timedelta(i * 7)).isoformat() \
        for i in range(8)]
output_root = '../../retrospective-hub/model-output'

commands = [f'python gbq.py --ref_date {ref_date} --output_root {output_root} --model_name gbq_qr_no_level' \
                for ref_date in missing_ref_dates]

with Pool(processes=2) as pool:
    pool.map(run_command, commands)
