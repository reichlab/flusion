import os
from pathlib import Path

from itertools import chain, product

import datetime
import time

import math
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

import datetime
import pymmwr

import lightgbm as lgb

# https://github.com/reichlab/timeseriesutils
from timeseriesutils import featurize

from data_pipeline.loader import FluDataLoader
from utils import create_features_and_targets



# config settings

# date of forecast generation
forecast_date = datetime.date.today()

# next Saturday: weekly forecasts are relative to this date
ref_date = forecast_date - datetime.timedelta((forecast_date.weekday() + 2) % 7 - 7)
print(f'reference date = {ref_date}')

# include measures of level among features?
incl_level_feats = False

# maximum forecast horizon
max_horizon = 5

# bagging setup
num_bags = 10
bag_frac_samples = 0.7

# quantile levels at which to generate predictions
q_levels = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
            0.85, 0.90, 0.95, 0.975, 0.99]
q_labels = ['0.01', '0.025', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35',
            '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8',
            '0.85', '0.9', '0.95', '0.975', '0.99']

# seed for random number generation, based on reference date
rng_seed = int(time.mktime(ref_date.timetuple()))
rng = np.random.default_rng(seed=rng_seed)
# seeds for lgb model fits, one per combination of bag and quantile level
lgb_seeds = rng.integers(1e8, size=(num_bags, len(q_levels)))


# load flu data
fdl = FluDataLoader('../../data-raw')
df = fdl.load_data()

# augment data with features and target values
df, feat_names = create_features_and_targets(
  df = df,
  incl_level_feats=incl_level_feats,
  max_horizon=max_horizon,
  curr_feat_names=['inc_4rt_cs', 'season_week', 'log_pop'])

# keep only rows that are in-season
df = df.query("season_week >= 5 and season_week <= 45")

# "test set" df used to generate look-ahead predictions
df_test = df \
    .loc[df.wk_end_date == df.wk_end_date.max()] \
    .copy()
x_test = df_test[feat_names]

# "train set" df for model fitting; target value non-missing
df_train = df.loc[~df['delta_target'].isna().values]
x_train = df_train[feat_names]
y_train = df_train['delta_target']


# training loop over bags
oob_preds_by_bag = np.empty((x_train.shape[0], num_bags, len(q_levels)))
oob_preds_by_bag[:] = np.nan
test_preds_by_bag = np.empty((x_test.shape[0], num_bags, len(q_levels)))

train_seasons = df_train['season'].unique()

for b in range(num_bags):
  print(f'bag number {b+1}')
  # get indices of observations that are in bag
  bag_seasons = rng.choice(
    train_seasons,
    size = int(len(train_seasons) * bag_frac_samples),
    replace=False)
  bag_obs_inds = df_train['season'].isin(bag_seasons)
  
  for q_ind, q_level in enumerate(q_levels):
    # fit to bag
    model = lgb.LGBMRegressor(
      verbosity=-1,
      objective='quantile',
      alpha=q_level,
      random_state=lgb_seeds[b, q_ind])
    model.fit(X=x_train.loc[bag_obs_inds, :], y=y_train.loc[bag_obs_inds])
    
    # oob predictions and test set predictions
    oob_preds_by_bag[~bag_obs_inds, b, q_ind] = model.predict(X=x_train.loc[~bag_obs_inds, :])
    test_preds_by_bag[:, b, q_ind] = model.predict(X=x_test)


# combined predictions across bags: median
oob_pred_qs = np.nanmedian(oob_preds_by_bag, axis=1)
test_pred_qs = np.median(test_preds_by_bag, axis=1)
test_pred_qs.shape

# test predictions as a data frame, one column per quantile level
test_pred_qs_df = pd.DataFrame(test_pred_qs)
test_pred_qs_df.columns = q_labels

# add predictions to original test df
df_test.reset_index(drop=True, inplace=True)
df_test_w_preds = pd.concat([df_test, test_pred_qs_df], axis=1)

# melt to get columns into rows, keeping only the things we need to invert data
# transforms later on
cols_to_keep = ['source', 'location', 'wk_end_date', 'pop',
                'inc_4rt_cs', 'horizon',
                'inc_4rt_center_factor', 'inc_4rt_scale_factor']
preds_df = df_test_w_preds[cols_to_keep + q_labels]
preds_df = preds_df.loc[(preds_df['source'] == 'hhs')]
preds_df = pd.melt(preds_df,
                   id_vars=cols_to_keep,
                   var_name='quantile',
                   value_name = 'delta_hat')

# build data frame with predictions on the original scale
preds_df['inc_4rt_cs_target_hat'] = preds_df['inc_4rt_cs'] + preds_df['delta_hat']
preds_df['inc_4rt_target_hat'] = (preds_df['inc_4rt_cs_target_hat'] + preds_df['inc_4rt_center_factor']) * (preds_df['inc_4rt_scale_factor'] + 0.01)
preds_df['value'] = (np.maximum(preds_df['inc_4rt_target_hat'], 0.0) ** 4 - 0.01 - 0.75**4) * preds_df['pop'] / 100000
preds_df['value'] = np.maximum(preds_df['value'], 0.0)

# keep just required columns and rename to match hub format
preds_df = preds_df[['location', 'wk_end_date', 'horizon', 'quantile', 'value']] \
    .rename(
        columns={
            'quantile': 'output_type_id'
        })

preds_df['target_end_date'] = preds_df['wk_end_date'] + pd.to_timedelta(7*preds_df['horizon'], unit='days')
preds_df['reference_date'] = ref_date
preds_df['horizon'] = preds_df['horizon'] - 2
preds_df['target'] = 'wk inc flu hosp'

preds_df['output_type'] = 'quantile'
preds_df.drop(columns='wk_end_date', inplace=True)

# sort quantiles to avoid quantile crossing
gcols = ['location', 'reference_date', 'horizon', 'target_end_date', 'target', 'output_type']
g = preds_df.set_index(gcols).groupby(gcols)
preds_df = g[['output_type_id', 'value']] \
  .transform(lambda x: x.sort_values()) \
  .reset_index()

# save
if not Path('../../submissions-hub/model-output/UMass-gbq_qr_no_level').exists():
    Path('../../submissions-hub/model-output/UMass-gbq_qr_no_level').mkdir(parents=True)

preds_df.to_csv(f'../../submissions-hub/model-output/UMass-gbq_qr_no_level/{str(ref_date)}-UMass-gbq_qr_no_level_refactored_run2.csv', index=False)


