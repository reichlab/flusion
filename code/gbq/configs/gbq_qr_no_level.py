import copy
from configs.base import base_config

config = copy.deepcopy(base_config)
config.model_name = 'gbq_qr_no_level'
config.incl_level_feats = False
