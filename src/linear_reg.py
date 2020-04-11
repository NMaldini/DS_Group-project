import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing

from src.constants import *

train_set = pd.read_csv(TRANING_DATA_LOCATION)
test_set = pd.read_csv(TEST_DATA_LOCATION)

selected_predictors = [
    'recharge_value',
    'data_mb',
    'voice_balance',
    'data_balance',
    'moc_same_network_min',
    'day_mou_min',
    'time_since_last_recharge',

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
    coeff_df = pd.DataFrame(regressor.coef_, selected_predictors, columns=['Coefficient'])
    print(coeff_df)
    print("intercept                   ", regressor.intercept_)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("Total actual revenue :  ", sum(df['Actual']))
    print("Total revenue :  ", sum(df['Predicted']))
    return regressor, sum(df['Predicted'])


def predict(model, X_test, y_test):
    regressor = model
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    coeff_df = pd.DataFrame(regressor.coef_, selected_predictors, columns=['Coefficient'])
    print("Total revenue after promotions :  ", sum(df['Predicted']))
    return sum(df['Predicted'])


model, r_1 = sklearn_model(X_train, y_train, X_test, y_test)

# Promotions
data = pd.DataFrame(test_set[['data_mb', 'moc_same_network_min']])
x = data.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
scaled_data = pd.DataFrame(x_scaled)
scaled_data.columns = ['data_mb', 'moc_same_network_min']
prop = 0.02

test_set['data_mb_scaled'] = scaled_data['data_mb']
test_set['moc_same_network_min_scaled'] = scaled_data['moc_same_network_min']
test_set['data_mb'] = np.where(test_set['data_mb_scaled'] >= prop, test_set['data_mb'] + 1000, test_set['data_mb'])
test_set['moc_same_network_min'] = np.where(test_set['moc_same_network_min_scaled'] >= prop,
                                            test_set['moc_same_network_min'] + 65, test_set['moc_same_network_min'])
test_set['recharge_value'] = np.where(test_set['moc_same_network_min_scaled'] >= prop,
                                      test_set['recharge_value'] + 50, test_set['recharge_value'])
test_set['recharge_value'] = np.where(test_set['data_mb_scaled'] >= prop,
                                      test_set['recharge_value'] + 49, test_set['recharge_value'])

X_test = test_set[selected_predictors]
r_2 = predict(model, X_test, y_test)
print("Revenue uplift : ", r_2 - r_1)
print("count of customers who got data offers : ", len(test_set[(test_set['data_mb_scaled'] >= prop)]))
print("count of customers who got voice offers : ", len(test_set[(test_set['moc_same_network_min_scaled'] >= prop)]))
