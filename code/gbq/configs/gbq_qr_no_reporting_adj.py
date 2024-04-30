import copy
from configs.base import base_config

config = copy.deepcopy(base_config)
config.model_name = 'gbq_qr_no_reporting_adj'

config.reporting_adj = False
