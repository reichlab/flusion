from pathlib import Path

from itertools import product

import datetime

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

import datetime
import pymmwr

from . import utils

class FluDataLoader():
  def __init__(self, data_raw) -> None:
    self.data_raw = Path(data_raw)


  def load_fips_mappings(self):
    return pd.read_csv(self.data_raw / 'fips-mappings/fips_mappings.csv')


  def load_flusurv_rates_2022_23(self):
    dat = pd.read_csv(self.data_raw / 'influenza-flusurv/flusurv-rates/flusurv-rates-2022-23.csv',
                      encoding='ISO-8859-1',
                      engine='python')
    dat.columns = dat.columns.str.lower()
    
    dat = dat.loc[(dat['age category'] == 'Overall') &
                  (dat['sex category'] == 'Overall') &
                  (dat['race category'] == 'Overall')]
    
    dat = dat.loc[~((dat.catchment == 'Entire Network') &
                    (dat.network != "FluSurv-NET"))]

    dat['location'] = dat['catchment']
    dat['agg_level'] = np.where(dat['location'] == 'Entire Network', 'national', 'site')
    dat['season'] = dat['year'].str.replace('-', '/')
    epiweek = dat['mmwr-year'].astype(str) + dat['mmwr-week'].astype(str)
    dat['season_week'] = utils.convert_epiweek_to_season_week(epiweek)
    dat['wk_end_date'] = dat.apply(
      lambda x: pymmwr.epiweek_to_date(pymmwr.Epiweek(year=x['mmwr-year'],
                                                      week=x['mmwr-week'],
                                                      day=7))
                                      .strftime("%Y-%m-%d"),
        axis=1)
    dat['wk_end_date'] = pd.to_datetime(dat['wk_end_date'])
    dat['inc'] = dat['weekly rate ']
    dat = dat[['agg_level', 'location', 'season', 'season_week', 'wk_end_date', 'inc']]
    
    return dat


  def load_flusurv_rates_base(self, 
                              seasons=None,
                              locations=['California', 'Colorado', 'Connecticut', 'Entire Network',
                                        'Georgia', 'Maryland', 'Michigan', 'Minnesota', 'New Mexico',
                                        'New York - Albany', 'New York - Rochester', 'Ohio', 'Oregon',
                                        'Tennessee', 'Utah'],
                              age_labels=['0-4 yr', '5-17 yr', '18-49 yr', '50-64 yr', '65+ yr', 'Overall']
                              ):
    # read flusurv data and do some minimal preprocessing
    dat = pd.read_csv(self.data_raw / 'influenza-flusurv/flusurv-rates/old-flusurv-rates.csv',
                      encoding='ISO-8859-1',
                      engine='python')
    dat.columns = dat.columns.str.lower()
    dat['season'] = dat.sea_label.str.replace('-', '/')
    dat['inc'] = dat.weeklyrate
    dat['location'] = dat['region']
    dat['agg_level'] = np.where(dat['location'] == 'Entire Network', 'national', 'site')
    dat = dat[(dat.age_label.isin(age_labels)) & (dat.location.isin(locations))]
    if seasons is not None:
      dat = dat[dat.season.isin(seasons)]
    
    dat = dat.sort_values(by=['wk_end'])
    
    dat['wk_end_date'] = pd.to_datetime(dat['wk_end'])
    dat = dat[['agg_level', 'location', 'season', 'season_week', 'wk_end_date', 'inc']]
    
    dat = pd.concat(
      [dat, self.load_flusurv_rates_2022_23()],
      axis = 0
    )

    dat['source'] = 'flusurvnet'
    
    return dat


  def load_one_us_census_file(self, f):
    dat = pd.read_csv(f, engine='python', dtype={'STATE': str})
    dat = dat.loc[(dat['NAME'] == 'United States') | (dat['STATE'] != '00'),
                  (dat.columns == 'STATE') | (dat.columns.str.startswith('POPESTIMATE'))]
    dat = dat.melt(id_vars = 'STATE', var_name='season', value_name='pop')
    dat.rename(columns={'STATE': 'location'}, inplace=True)
    dat.loc[dat['location'] == '00', 'location'] = 'US'
    dat['season'] = dat['season'].str[-4:]
    dat['season'] = dat['season'] + '/' + (dat['season'].str[-2:].astype(int) + 1).astype(str)
    
    return dat


  def load_us_census(self, fillna = True):
    files = [
      self.data_raw / 'us-census/nst-est2019-alldata.csv',
      self.data_raw / 'us-census/NST-EST2022-ALLDATA.csv']
    us_pops = pd.concat([self.load_one_us_census_file(f) for f in files], axis=0)
    
    fips_mappings = pd.read_csv(self.data_raw / 'fips-mappings/fips_mappings.csv')
    
    hhs_pops = us_pops.query("location != 'US'") \
      .merge(
          fips_mappings.query("location != 'US'") \
              .assign(hhs_region=lambda x: 'Region ' + x['hhs_region'].astype(int).astype(str)),
          on='location',
          how = 'left'
      ) \
      .groupby(['hhs_region', 'season']) \
      ['pop'] \
      .sum() \
      .reset_index() \
      .rename(columns={'hhs_region': 'location'})
    
    dat = pd.concat([us_pops, hhs_pops], axis=0)
    
    if fillna:
      all_locations = dat['location'].unique()
      all_seasons = [str(y) + '/' + str(y+1)[-2:] for y in range(1997, 2024)]
      full_result = pd.DataFrame.from_records(product(all_locations, all_seasons))
      full_result.columns = ['location', 'season']
      dat = full_result.merge(dat, how='left', on=['location', 'season']) \
        .set_index('location') \
        .groupby(['location']) \
        .bfill() \
        .groupby(['location']) \
        .ffill() \
        .reset_index()
    
    return dat


  def load_hosp_burden(self):
    burden_estimates = pd.read_csv(
      self.data_raw / 'burden-estimates/burden-estimates.csv',
      engine='python')

    burden_estimates.columns = ['season', 'hosp_burden']

    #burden_estimates['hosp_burden'] = burden_estimates['hosp_burden'].astype(int)

    burden_estimates['season'] = burden_estimates['season'].str[:4] + '/' + burden_estimates['season'].str[7:9]

    return burden_estimates


  def calc_hosp_burden_adj(self):
    dat = self.load_flusurv_rates_base(
      seasons = ['20' + str(yy) + '/' + str(yy+1) for yy in range(10, 23)],
      locations= ['Entire Network'],
      age_labels = ['Overall']
    )

    burden_adj = dat[dat.location == 'Entire Network'] \
      .groupby('season')['inc'] \
      .sum()
    burden_adj = burden_adj.reset_index()
    burden_adj.columns = ['season', 'cum_rate']

    us_census = self.load_us_census().query("location == 'US'").drop('location', axis=1)
    burden_adj = pd.merge(burden_adj, us_census, on='season')

    burden_estimates = self.load_hosp_burden()
    burden_adj = pd.merge(burden_adj, burden_estimates, on='season')

    burden_adj['reported_burden_est'] = burden_adj['cum_rate'] * burden_adj['pop'] / 100000
    burden_adj['adj_factor'] = burden_adj['hosp_burden'] / burden_adj['reported_burden_est']

    return burden_adj


  def fill_missing_flusurv_dates_one_location(self, location_df):
    df = location_df.set_index('wk_end_date') \
      .asfreq('W-sat') \
      .reset_index()
    fill_cols = ['agg_level', 'location', 'season', 'pop', 'source']
    fill_cols = [c for c in fill_cols if c in df.columns]
    df[fill_cols] = df[fill_cols].fillna(axis=0, method='ffill')
    return df


  def load_flusurv_rates(self,
                         burden_adj=True,
                         locations=['California', 'Colorado', 'Connecticut', 'Entire Network',
                                    'Georgia', 'Maryland', 'Michigan', 'Minnesota', 'New Mexico',
                                    'New York - Albany', 'New York - Rochester', 'Ohio', 'Oregon',
                                    'Tennessee', 'Utah']
                        ):
    # read flusurv data and do some minimal preprocessing
    dat = self.load_flusurv_rates_base(
      seasons = ['20' + str(yy) + '/' + str(yy+1) for yy in range(10, 23)],
      locations = locations,
      age_labels = ['Overall']
    )
    
    # if requested, make adjustments for overall season burden
    if burden_adj:
      hosp_burden_adj = self.calc_hosp_burden_adj()
      dat = pd.merge(dat, hosp_burden_adj, on='season')
      dat['inc'] = dat['inc'] * dat['adj_factor']
    
    # fill in missing dates
    gd = dat.groupby('location')
    
    dat = pd.concat(
      [self.fill_missing_flusurv_dates_one_location(df) for _, df in gd],
      axis = 0)
    dat = dat[['agg_level', 'location', 'season', 'season_week', 'wk_end_date', 'inc', 'source']]
    
    return dat


  def load_who_nrevss_positive(self):
    dat = pd.read_csv(self.data_raw / 'influenza-who-nrevss/who-nrevss.csv',
                      encoding='ISO-8859-1',
                      engine='python')
    dat = dat[['region_type', 'region', 'year', 'week', 'season', 'season_week', 'percent_positive']]
    
    dat.rename(columns={'region_type': 'agg_level', 'region': 'location'},
              inplace=True)
    dat['agg_level'] = np.where(dat['agg_level'] == 'National',
                                'national',
                                dat['agg_level'].str[:-1].str.lower())
    return dat


  def load_ilinet(self,
                  response_type='rate',
                  scale_to_positive=True,
                  drop_pandemic_seasons=True,
                  burden_adj=False):
    # read ilinet data and do some minimal preprocessing
    files = [self.data_raw / 'influenza-ilinet/ilinet.csv',
             self.data_raw / 'influenza-ilinet/ilinet_hhs.csv',
             self.data_raw / 'influenza-ilinet/ilinet_state.csv']
    dat = pd.concat(
      [ pd.read_csv(f, encoding='ISO-8859-1', engine='python') for f in files ],
      axis = 0)
    
    if response_type == 'rate':
      dat['inc'] = np.where(dat['region_type'] == 'States',
                            dat['unweighted_ili'],
                            dat['weighted_ili'])
    else:
      dat['inc'] = dat.ilitotal

    dat['wk_end_date'] = pd.to_datetime(dat['week_start']) + pd.Timedelta(6, 'days')
    dat = dat[['region_type', 'region', 'year', 'week', 'season', 'season_week', 'wk_end_date', 'inc']]
    
    dat.rename(columns={'region_type': 'agg_level', 'region': 'location'},
              inplace=True)
    dat['agg_level'] = np.where(dat['agg_level'] == 'National',
                                'national',
                                dat['agg_level'].str[:-1].str.lower())
    dat = dat.sort_values(by=['season', 'season_week'])
    
    # for early seasons, drop out-of-season weeks with no reporting
    early_seasons = [str(yyyy) + '/' + str(yyyy + 1)[2:] for yyyy in range(1997, 2002)]
    early_in_season_weeks = [w for w in range(10, 43)]
    first_report_season = ['2002/03']
    first_report_in_season_weeks = [w for w in range(10, 53)]
    dat = dat[
      (dat.season.isin(early_seasons) & dat.season_week.isin(early_in_season_weeks)) |
      (dat.season.isin(first_report_season) & dat.season_week.isin(first_report_in_season_weeks)) |
      (~dat.season.isin(early_seasons + first_report_season))]
    
    # region 10 data prior to 2010/11 is bad, drop it
    dat = dat[
      ~((dat['location'] == 'Region 10') & (dat['season'] < '2010/11'))
    ]
    
    if scale_to_positive:
      dat = pd.merge(
        left=dat,
        right=self.load_who_nrevss_positive(),
        how='left',
        on=['agg_level', 'location', 'season', 'season_week'])
      dat['inc'] = dat['inc'] * dat['percent_positive'] / 100.0
      dat.drop('percent_positive', axis=1)

    if drop_pandemic_seasons:
      dat.loc[dat['season'].isin(['2008/09', '2009/10', '2020/21', '2021/22']), 'inc'] = np.nan

    # if requested, make adjustments for overall season burden
    # if burden_adj:
    #   hosp_burden_adj = calc_hosp_burden_adj()
    #   dat = pd.merge(dat, hosp_burden_adj, on='season')
    #   dat['inc'] = dat['inc'] * dat['adj_factor']

    dat = dat[['agg_level', 'location', 'season', 'season_week', 'wk_end_date', 'inc']]
    dat['source'] = 'ilinet'
    return dat


  def load_hhs(self, rates=True):
    dat = pd.read_csv(self.data_raw / 'influenza-hhs/hhs.csv')
    dat.rename(columns={'date': 'wk_end_date'}, inplace=True)

    ew_str = dat.apply(utils.date_to_ew_str, axis=1)
    dat['season'] = utils.convert_epiweek_to_season(ew_str)
    dat['season_week'] = utils.convert_epiweek_to_season_week(ew_str)
    dat = dat.sort_values(by=['season', 'season_week'])
    
    if rates:
      pops = self.load_us_census()
      dat = dat.merge(pops, on = ['location', 'season'], how='left') \
        .assign(inc=lambda x: x['inc'] / x['pop'] * 100000)

    dat['wk_end_date'] = pd.to_datetime(dat['wk_end_date'])
    
    dat['agg_level'] = np.where(dat['location'] == 'US', 'national', 'state')
    dat = dat[['agg_level', 'location', 'season', 'season_week', 'wk_end_date', 'inc']]
    dat['source'] = 'hhs'
    return dat
