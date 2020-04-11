DATA_SET_LOCATION = '../data/Dataset.xlsx'
DATA_DICT_LOCATION = '../DS_Group_Data Dictionary (1).xlsx'
TRANING_DATA_LOCATION = '../data/train.csv'
TEST_DATA_LOCATION = '../data/test.csv'
DATA_DICT_SHEET = 'DataSet'
DATA_TYPES = ['Categorical Nominal', 'Metric Discrete', 'Metric Continuous' ]

PREDICTORS = [
    'recharge_value',
    # 'voice_revenue',
    # 'data_revenue',
    'data_mb',
    # 'rc_slab_30',
    # 'time_since_last_recharge',
    'voice_balance',
    # 'data_balance',
    # 'network_stay',
    # 'mtc_idd_min',
    # 'moc_same_network_min',
    # 'mtc_same_network_min',
    # 'moc_idd_min',
    # 'total_moc_count',
    # 'mtc_other_networks'
    'mtc_major',
    'rc_slab_100',
    # 'total_og_min',
    'moc_other_networks',
    'last_rec_denom',
    'moc_major',
    # 'rc_slab_50',
    'day_mou_min'
    # rc_slab_59
    # language
    # night_mou_min
    # time_since_last_data_use
    # rc_slab_119
    # time_since_last_call
    # time_since_last_activity
    # smart_ph_flag
    # rc_slab_99
    # rc_slab_49
    # dual_sim_flag
]