import copy
from configs.base import base_config

config = copy.deepcopy(base_config)
config.model_name = 'gbq_qr_fit_locations_separately'
config.fit_locations_separately = True
