from data_pipeline.loader import FluDataLoader

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
