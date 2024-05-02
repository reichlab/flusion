import pytest
from data_pipeline.loader import FluDataLoader
import numpy as np
import datetime

def test_load_data_sources():
    fdl = FluDataLoader('../../data-raw')
    
    sources_options = [
        ['hhs'],
        ['hhs', 'ilinet'],
        ['flusurvnet'],
        ['flusurvnet', 'hhs', 'ilinet']
    ]
    for sources in sources_options:
        df = fdl.load_data(sources=sources)
        assert set(df['source'].unique()) == set(sources)
    
    df = fdl.load_data()
    assert set(df['source'].unique()) == {'flusurvnet', 'hhs', 'ilinet'}


@pytest.mark.parametrize("test_kwargs, season_expected, wk_end_date_expected", [
    (None, '2022/23', '2023-12-23'),
    ({'drop_pandemic_seasons': False}, '2019/20', '2023-12-23'),
    ({'drop_pandemic_seasons': True, 'as_of': datetime.date.fromisoformat('2023-12-30')},
        '2022/23', '2023-12-23')
])
def test_load_data_hhs_kwargs(test_kwargs, season_expected, wk_end_date_expected):
    fdl = FluDataLoader('../../data-raw')
    
    df = fdl.load_data(sources=['hhs'], hhs_kwargs=test_kwargs)
    
    assert df['season'].min() == season_expected
    wk_end_date_actual = str(df['wk_end_date'].max())[:10]
    if test_kwargs is not None and 'as_of' in test_kwargs:
        assert wk_end_date_actual == wk_end_date_expected
    else:
        assert wk_end_date_actual > wk_end_date_expected


@pytest.mark.parametrize("test_kwargs, expect_all_na", [
    (None, True),
    ({'drop_pandemic_seasons': False}, False),
    ({'drop_pandemic_seasons': True}, True)
])
def test_load_data_ilinet_kwargs(test_kwargs, expect_all_na):
    fdl = FluDataLoader('../../data-raw')
    
    df = fdl.load_data(sources=['ilinet'], ilinet_kwargs=test_kwargs)
    
    if expect_all_na:
        assert np.all(df.loc[df['season'].isin(['2008/09', '2009/10', '2020/21', '2021/22']), 'inc'].isna())
    else:
        # expect some non-NA values in pandemic seasons
        assert np.any(~df.loc[df['season'].isin(['2008/09', '2009/10', '2020/21', '2021/22']), 'inc'].isna())


@pytest.mark.parametrize("test_kwargs", [
    (None),
    ({'locations': ['California', 'Colorado', 'Connecticut']})
])
def test_load_data_flusurvnet_kwargs(test_kwargs):
    fdl = FluDataLoader('../../data-raw')
    
    #flusurv_kwargs
    df = fdl.load_data(sources=['flusurvnet'], flusurvnet_kwargs=test_kwargs)
    
    if test_kwargs is None:
        assert len(df['location'].unique()) > 3
    else:
        assert len(df['location'].unique()) == len(test_kwargs['locations'])
