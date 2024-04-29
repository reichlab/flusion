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


def test_load_data_kwargs():
    fdl = FluDataLoader('../../data-raw')
    
    # hhs_kwargs
    df_hhs1 = fdl.load_data(sources=['hhs'])
    df_hhs2 = fdl.load_data(
        sources=['hhs'],
        hhs_kwargs={
            'drop_pandemic_seasons': False
        })
    df_hhs3 = fdl.load_data(
        sources=['hhs'],
        hhs_kwargs={
            'drop_pandemic_seasons': True,
            'as_of': datetime.date.fromisoformat('2023-12-30')
        })

    assert df_hhs1['season'].min() == '2022/23'
    assert df_hhs2['season'].min() == '2019/20'
    assert df_hhs3['season'].min() == '2022/23'
    assert str(df_hhs1['wk_end_date'].max())[:10] > '2023-12-23'
    assert str(df_hhs2['wk_end_date'].max())[:10] > '2023-12-23'
    assert str(df_hhs3['wk_end_date'].max())[:10] == '2023-12-23'
    
    # ilinet_kwargs
    df_ilinet1 = fdl.load_data(sources=['ilinet'])
    df_ilinet2 = fdl.load_data(
        sources=['ilinet'],
        ilinet_kwargs={'drop_pandemic_seasons': True})
    df_ilinet3 = fdl.load_data(
        sources=['ilinet'],
        ilinet_kwargs={'drop_pandemic_seasons': False})
    
    # in first two results, pandemic season incidence set to NA
    assert np.all(df_ilinet1.loc[df_ilinet1['season'].isin(['2008/09', '2009/10', '2020/21', '2021/22']), 'inc'].isna())
    assert np.all(df_ilinet2.loc[df_ilinet2['season'].isin(['2008/09', '2009/10', '2020/21', '2021/22']), 'inc'].isna())
    # in third result, some non-NA values in pandemic seasons
    assert np.any(~df_ilinet3.loc[df_ilinet1['season'].isin(['2008/09', '2009/10', '2020/21', '2021/22']), 'inc'].isna())
    
    #flusurv_kwargs
    df_flusurv1 = fdl.load_data(sources=['flusurvnet'])
    df_flusurv2 = fdl.load_data(
        sources=['flusurvnet'],
        flusurvnet_kwargs={'locations': ['California', 'Colorado', 'Connecticut']})
    
    assert len(df_flusurv1['location'].unique()) > 3
    assert len(df_flusurv2['location'].unique()) == 3
