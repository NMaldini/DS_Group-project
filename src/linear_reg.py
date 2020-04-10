import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm

from src.constants import *

train_set = pd.read_csv(TRANING_DATA_LOCATION)
test_set = pd.read_csv(TEST_DATA_LOCATION)

selected_predictors = [
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
X_train = train_set[selected_predictors]
y_train = train_set['revenue_rs']
X_test = test_set[selected_predictors]
y_test = test_set['revenue_rs']


def sklearn_model(_train, y_train, X_test, y_test):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    print(df1)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    coeff_df = pd.DataFrame(regressor.coef_, selected_predictors, columns=['Coefficient'])


def stats_model_regressor(X_train, y_train, X_test, y_test):
    model = sm.OLS(y_train, X_train).fit()
    y_pred = model.predict(X_test)  # make the predictions by the model
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    print(df1)
    # Print out the statistics
    print(model.summary())



stats_model_regressor(X_train, y_train, X_test, y_test)
