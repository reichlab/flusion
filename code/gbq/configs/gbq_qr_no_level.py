import copy
from configs.base import base_config

config = copy.deepcopy(base_config)
config.incl_level_feats = False
