import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace

import datetime

import pandas as pd

from timeseriesutils import featurize
from data_pipeline.utils import get_holidays

def make_parser():
    parser = argparse.ArgumentParser(description='Run gradient boosting model for flu prediction')
    parser.add_argument('--ref_date',
                        help='reference date for predictions in format YYYY-MM-DD; a Saturday',
                        type=lambda s: datetime.date.fromisoformat(s),
                        default=None)
    parser.add_argument('--model_name',
                        help='Model name',
                        choices=['gbq_qr', 'gbq_qr_no_level'],
                        default='gbq_qr')
    parser.add_argument('--short_run',
                        help='Flag to do a short run; overrides model-default num_bags to 10 and uses 3 quantile levels',
                        action='store_true')
    parser.add_argument('--output_root',
                        help='Path to a directory in which model outputs are saved',
                        type=lambda s: Path(s),
                        default=Path('../../submissions-hub/model-output'))
    
    return parser


def validate_ref_date(ref_date):
    if ref_date is None:
        today = datetime.date.today()
        
        # next Saturday: weekly forecasts are relative to this date
        ref_date = today - datetime.timedelta((today.weekday() + 2) % 7 - 7)
        
        return ref_date
    elif isinstance(ref_date, datetime.date):
        # check that it's a Saturday
        if ref_date.weekday() != 5:
            raise ValueError('ref_date must be a Saturday')
        
        return ref_date
    else:
        raise TypeError('ref_date must be a datetime.date object')


def parse_args():
    '''
    Parse arguments to the gbq_qr.py script
    
    Returns
    -------
    Two dictionaries collecting settings for the model and the run:
    - `model_config` contains settings for the model
    - `run_config` contains the following properties:
        - `ref_date`: the reference date for the forecast
        - `output_root`: `pathlib.Path` object with the root directory for
            saving model outputs
        - `max_horizon`: integer, maximum forecast horizon relative to the
            last observed data
        - `q_levels`: list of floats with quantile levels for predictions
        - `q_labels`: list of strings with names for the quantile levels
    '''
    parser = make_parser()
    args = parser.parse_args()
    
    ref_date = validate_ref_date(args.ref_date)
    model_name = args.model_name
    
    model_config = importlib.import_module(f'configs.{model_name}').config
    
    run_config = SimpleNamespace(
        ref_date=ref_date,
        output_root=args.output_root
    )
    
    if args.short_run:
        # override model-specified num_bags to a smaller value
        model_config.num_bags = 10
        
        # maximum forecast horizon
        run_config.max_horizon = 3
        
        # quantile levels at which to generate predictions
        run_config.q_levels = [0.025, 0.50, 0.975]
        run_config.q_labels = ['0.025', '0.5', '0.975']
    else:
        # maximum forecast horizon
        run_config.max_horizon = 5
        
        # quantile levels at which to generate predictions
        run_config.q_levels = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                               0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                               0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
        run_config.q_labels = ['0.01', '0.025', '0.05', '0.1', '0.15', '0.2',
                               '0.25', '0.3', '0.35', '0.4', '0.45', '0.5',
                               '0.55', '0.6', '0.65', '0.7', '0.75', '0.8',
                               '0.85', '0.9', '0.95', '0.975', '0.99']
    
    return model_config, run_config


def create_features_and_targets(df, incl_level_feats, max_horizon, curr_feat_names = []):
    '''
    Create features and targets for prediction
    
    Parameters
    ----------
    df: pandas dataframe
      data frame with data to "featurize"
    incl_level_feats: boolean
      include features that are a measure of local level of the signal?
    max_horizon: int
      maximum forecast horizon
    curr_feat_names: list of strings
      list of names of columns in `df` containing existing features
    
    Returns
    -------
    tuple with:
    - the input data frame, augmented with additional columns with feature and
      target values
    - a list of all feature names, columns in the data frame
    '''
    
    # current features; will be updated
    feat_names = curr_feat_names
    
    # one-hot encodings of data source, agg_level, and location
    for c in ['source', 'agg_level', 'location']:
        ohe = pd.get_dummies(df[c], prefix=c)
        df = pd.concat([df, ohe], axis=1)
        feat_names = feat_names + list(ohe.columns)
    
    # season week relative to christmas
    df = df.merge(
            get_holidays() \
                .query("holiday == 'Christmas Day'") \
                .drop(columns=['holiday', 'date']) \
                .rename(columns={'season_week': 'xmas_week'}),
            how='left',
            on='season') \
        .assign(delta_xmas = lambda x: x['season_week'] - x['xmas_week'])
    
    feat_names = feat_names + ['delta_xmas']
    
    # features summarizing data within each combination of source and location
    df, new_feat_names = featurize.featurize_data(
        df, group_columns=['source', 'location'],
        features = [
            {
                'fun': 'windowed_taylor_coefs',
                'args': {
                    'columns': 'inc_4rt_cs',
                    'taylor_degree': 2,
                    'window_align': 'trailing',
                    'window_size': [4, 6],
                    'fill_edges': False
                }
            },
            {
                'fun': 'windowed_taylor_coefs',
                'args': {
                    'columns': 'inc_4rt_cs',
                    'taylor_degree': 1,
                    'window_align': 'trailing',
                    'window_size': [3, 5],
                    'fill_edges': False
                }
            },
            {
                'fun': 'rollmean',
                'args': {
                    'columns': 'inc_4rt_cs',
                    'group_columns': ['location'],
                    'window_size': [2, 4]
                }
            }
        ])
    feat_names = feat_names + new_feat_names
    
    df, new_feat_names = featurize.featurize_data(
        df, group_columns=['source', 'location'],
        features = [
            {
                'fun': 'lag',
                'args': {
                    'columns': ['inc_4rt_cs'] + new_feat_names,
                    'lags': [1, 2]
                }
            }
        ])
    feat_names = feat_names + new_feat_names
    
    # add forecast targets
    df, new_feat_names = featurize.featurize_data(
        df, group_columns=['source', 'location'],
        features = [
            {
                'fun': 'horizon_targets',
                'args': {
                    'columns': 'inc_4rt_cs',
                    'horizons': [(i + 1) for i in range(max_horizon)]
                }
            }
        ])
    feat_names = feat_names + new_feat_names
    
    # we will model the differences between the prediction target and the most
    # recent observed value
    df['delta_target'] = df['inc_4rt_cs_target'] - df['inc_4rt_cs']
    
    # if requested, drop features that involve absolute level
    if not incl_level_feats:
        level_feats = ['inc_4rt_cs', 'inc_4rt_cs_lag1', 'inc_4rt_cs_lag2'] + \
                      [f for f in feat_names if f.find('taylor_d0') > -1] + \
                      [f for f in feat_names if f.find('inc_4rt_cs_rollmean') > -1]
        feat_names = [f for f in feat_names if f not in level_feats]
    
    return df, feat_names
