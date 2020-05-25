from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import datetime
import time
import h2o
import pandas as pd
import numpy as np
from math import sin
from math import radians
#import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


#from h2o.estimators.gbm import H2OGradientBoostingEstimator
#from h2o.grid.grid_search import H2OGridSearch
#from h2o.estimators.glm import H2OGeneralizedLinearEstimator
#from h2o.estimators.random_forest import H2ORandomForestEstimator
#from h2o.estimators.deeplearning import H2ODeepLearningEstimator
#h2o.init(max_mem_size_GB=14)
#h2o.init(ip="localhost", port=54321)


cal = calendar()

targetCol = 'Quantity'


def calculate_forecast_errors(df):

    df = df.copy()

    df['e'] = df['y'] - df['yhat']
    df['p'] = 100 * df['e'] / df['y']

    #predicted_part = df[-prediction_size:]
    def error_mean(error_name): return np.mean(np.abs(df[error_name]))

    return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}


def createFeatures(df, label=None, calendar=[]):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    df['isholiday'] = df['date'].isin(calendar)
    df['isholiday'] = df['isholiday'].apply(lambda x: 1 if x == 'True' else 0)
    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear', 'isholiday']]
    # else:
    #    X = df[['hour','dayofweek','quarter','month','year',
    #        'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

#define function for kpss test
from statsmodels.tsa.stattools import kpss
#define KPSS
def kpss_test(timeseries, quantity):
    print ('Results of KPSS Test:')
    kpss_output = {}
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:1], index=[quantity])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    return kpss_output

#define function for ADF test
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries, quantity):
    #Perform Dickey-Fuller test:
    dfoutput = {}
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:1], index=[quantity])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    return dfoutput

# create a differenced series
def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
                value = dataset[i] - dataset[i - interval]
                diff.append(value)
        return diff

# invert differenced forecast
def inverse_difference(last_ob, value):
        return value + last_ob

#check stationary
def makeStationary(dataset=None):
    kpss = kpss_test(dataset[targetCol], targetCol)
    if kpss.get(targetCol, 0) < kpss.get('Critical Value (1%)', 0):
	is_stationary_kpps = True
    else: is_stationary_kpps = False
    adf = adf_test(dataset[targetCol], targetCol)
    if adf.get(targetCol, 0) < adf.get('Critical Value (1%)', 0):
	is_stationary_adf = True
    else: is_stationary_adf = False
    if not is_stationary_kpps and not is_stationary_adf:
	diff = difference(dataset[targetCol], 1)
   	data = [inverse_difference(dataset[targetCol][i], diff[i]) for i in range(len(diff))]
    	dataset[targetCol] = [0] + data
    return dataset

def createTestDataFrame(trainData=None, days=10):
    days = pd.date_range(
        start=str(trainData['date'].iloc[-1]), periods=int(days))
    dates = days.to_pydatetime()
    dates = [str(i.strftime('%Y-%m-%d')) for i in dates]
    df = pd.DataFrame({'date': dates})
    df.to_csv('test_frame.csv', index=False)
    df = pd.read_csv('test_frame.csv', index_col=[0], parse_dates=[0])
    return df

'''
def h2oGBE(trainData=None, testData=None, targetColumn=None, featuresColumns=[]):
    if not targetColumn:
        targetColumn = targetCol
    gbm = H2OGradientBoostingEstimator(sample_rate=.7, seed=1234)
    trainData, featuresColumns, validData = prepareCols(
        trainData=trainData, featuresColumns=featuresColumns)
    #trainData, textData, validData = trainData.split_frame(ratios = [.7, .15], seed = 1234)
    gbm.train(x=featuresColumns, y=targetColumn,
              training_frame=trainData,
              validation_frame=validData)
    predict = gbm.predict(testData)
    submission = testData.concat(predict).as_data_frame(use_pandas=True)
    #submission['Date'] = pd.to_datetime(submission['Date']/1000)
    submission.to_csv('submission.csv', index=False)
    return submission

'''

def dataTuning(trainData=None, days=None, splitFalg=None, testData=None):
    trainData.to_csv('temp.csv', index=False)
    trainData = pd.read_csv('temp.csv', index_col=[0], parse_dates=[0])
    trainData = makeStationary(dataset=trainData)
    testData.to_csv('temp_test.csv', index=False)
    testData = pd.read_csv('temp_test.csv', index_col=[0], parse_dates=[0])
    holidays = cal.holidays(start=trainData.index.min(),
                            end=trainData.index.max())
    test_holidays = cal.holidays(
        start=testData.index.min(), end=testData.index.max())
    # split_date = '04-Mar-2019'#'02-Jan-2018'
    if splitFalg:
        dataset_train, dataset_test = np.split(
            trainData, [int(.8*len(trainData))])
        X_train, y_train = createFeatures(
            dataset_train, label=targetCol, calendar=holidays)
        X_test, y_test = createFeatures(
            dataset_test, label=targetCol, calendar=holidays)
    else:
        X_train, y_train = createFeatures(
            trainData, label=targetCol, calendar=holidays)
        X_test, y_test = None, None
    #futureData = createTestDataFrame(trainData=trainData, days=days)
    Z_test = createFeatures(testData, calendar=test_holidays)
    return (X_train, y_train, X_test, y_test, Z_test, testData)


def RFR(trainData=None, days=None, prdData=None):

    X_train, y_train, X_test, y_test, Z_test, testData = dataTuning(
        trainData=trainData, days=days, testData=prdData)
    rfr = RandomForestRegressor(n_estimators=1000, random_state=1)
    rfr.fit(X_train, y_train)
    #prdData['Predicted'] = rfr.predict(Z_test)
    # return prdData
    testData['Predicted'] = rfr.predict(Z_test)
    return testData


def GBR(trainData=None, days=None, prdData=None):
    X_train, y_train, X_test, y_test, Z_test, testData = dataTuning(
        trainData=trainData, days=days, testData=prdData)
    gbrt = GradientBoostingRegressor(n_estimators=1000, random_state=2)
    #gbrt = GradientBoostingRegressor()

    gbrt.fit(X_train, y_train)
    #prdData['Predicted'] = gbrt.predict(Z_test)
    # return prdData
    testData['Predicted'] = gbrt.predict(Z_test)
    return testData


def XGBR(trainData=None, days=None, prdData=None):
    #trainData.to_csv('temp.csv', index=False)
    #pjme = pd.read_csv('temp.csv', index_col=[0], parse_dates=[0])
    # split_date = '04-Mar-2019'#'02-Jan-2018'
    #pjme_train = pjme.loc[pjme.index <= split_date].copy()
    #pjme_test = pjme.loc[pjme.index > split_date].copy()
    #index = len(pjme.index)/4
    #pjme_train = pjme.iloc[:, :int(index)]
    #pjme_test = pjme.iloc[:, :int(index):]
    #X_train, y_train = createFeatures(pjme_train, label=targetCol)
    #X_test, y_test = createFeatures(pjme_test, label=targetCol)
    X_train, y_train, X_test, y_test, Z_test, testData = dataTuning(
        trainData=trainData, days=days, splitFalg='true', testData=prdData)
    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False)  # Change verbose to True if you want to see it train

    #futureData = createTestDataFrame(trainData=trainData, days=days)
    #Z_test = createFeatures(futureData)
    #prdData['Prediction'] =  reg.predict(Z_test)
    testData['Prediction'] = reg.predict(Z_test)
    return testData


def prepareCols(trainData=None, featuresColumns=[]):
    if not featuresColumns:
        featuresColumns = [i for i in trainData.columns]
    #_ty = {key:'string' for key in featuresColumns}
    #trainData = h2o.H2OFrame(trainData, column_types=_ty)
    trainData["DATE"] = trainData["DATE"].apply(
        lambda x: str(x).replace("-", ""))
    trainData["DATE"] = trainData["DATE"].astype(int)
    trainData = h2o.H2OFrame(trainData)
    #trainData, testData, validData = trainData.split_frame(ratios = [.7, .15], seed = 1234)
    trainData, validData = trainData.split_frame(ratios=[.7], seed=1234)
    '''
    for i in featuresColumns:
        try:trainData[i]= trainData[i].asfactor()
        except:trainData[i]= trainData[i].asfactor()
    '''
    return (trainData, featuresColumns, validData)
    # return (trainData, featuresColumns, validData, testData)

'''
def h2oRFE(featuresColumns=[], targetColumn=None, trainData=None, testData=None):
    if not targetColumn:
        targetColumn = targetCol
    #trainData = h2o.H2OFrame(trainData)
    #trainData, testData = trainData.split_frame(ratios = [.8], seed = 1234)
    # testData = h2o.upload_file('/root/sampleSheet2.csv')#h2o.H2OFrame(testData)
    # rfm = H2ORandomForestEstimator(nfolds = 10,ntrees = 500, stopping_metric = "RMSE",stopping_rounds = 10,
    #                        stopping_tolerance = 0.005,seed=1234)#(balance_classes=True)
    rfm = H2ORandomForestEstimator(seed=1)
    trainData, featuresColumns, validData = prepareCols(
        trainData=trainData, featuresColumns=featuresColumns)
    #X_train, y_train, X_test, y_test, Z_test = dataTuning(trainData=trainData, days=days, splitFalg='true')

    rfm.train(x=featuresColumns, y=targetColumn,
              validation_frame=validData, training_frame=trainData)
    predict = rfm.predict(testData)
    submission = testData.concat(predict).as_data_frame(use_pandas=True)
    #submission['date'] = pd.to_datetime(submission['date']/1000)
    submission.to_csv('submission.csv', index=False)
    return submission


def h2oGLE(featuresColumns=[], targetColumn=None, trainData=None, testData=None):
    #trainData = h2o.H2OFrame(trainData)
    if not targetColumn:
        targetColumn = targetCol
    # if not testData:
    #	trainData, validData, testData = trainData.split_frame(ratios = [0.7, 0.15], seed = 1234)
    trainData, featuresColumns, validData = prepareCols(
        trainData=trainData, featuresColumns=featuresColumns)
    glm = H2OGeneralizedLinearEstimator(family='binomial')
    # if not featuresColumns:
    #    featuresColumns = trainData.columns
    glm.train(x=featuresColumns, y=targetColumn,
              training_frame=trainData, validation_frame=validData)

    predict = glm.predict(testData)
    submission = testData.concat(predict).as_data_frame(use_pandas=True)
    submission.to_csv('submission.csv', index=False)
    return submission

def generateTestDataFrame(trainData=None, days=10):
    days = pd.date_range(start=str(trainData['date'].iloc[-1]), periods=int(days))
    dates = days.to_pydatetime()
    dates = [str(i.strftime('%Y-%m-%d')) for i in dates]
    df = pd.DataFrame({'date':dates})
    df["date"] = df["date"].apply(lambda x: x.replace("-",""))
    df["date"]  = df["date"].astype(int)
    _data = h2o.H2OFrame(df)
    #for i in cols:
    #    _data[i]= _data[i].asfactor()
    return _data
'''


def generateTestDates(trainData=None, days=0, gname=None):
    freq_set = {'Day': 'd', 'Week': 'w', 'Month': 'm'}
    if freq_set.get(gname) == 'Month':
        days = int(days) * 30
    elif freq_set.get(gname) == 'Week':
        days = int(days) * 7
    days = pd.date_range(
        start=str(trainData['DATE'].iloc[-1]), periods=int(days))
    dates = days.to_pydatetime()
    dates = [str(i.strftime('%Y-%m-%d')) for i in dates]
    return dates


def generateTestDataFrame(trainData=None, days=10, gname=None):
    data = trainData.groupby(['SKU', 'CATEGORY']).size().reset_index()
    dates = generateTestDates(trainData=trainData, days=days, gname=gname)
    list_ = []
    for date in dates:
        for row in data.iterrows():
            try:
                list_.append([date, row[1]['SKU'], row[1]['CATEGORY']])
            except:
                continue
    df = pd.DataFrame(list_, columns=['DATE', 'SKU', 'CATEGORY'])
    df["DATE"] = df["DATE"].apply(lambda x: x.replace("-", ""))
    df["DATE"] = df["DATE"].astype(int)
    dFrame = h2o.H2OFrame(df)
    return dFrame


def dataColumns(df=None):
    return df.columns


def cleanDataSet(df=None):
    # Identify the columns containing NA values in both train and test dataset
    df.isnull().sum().sort_values(ascending=False)
    # Identify the columns which contain numeric values
    numericCol = df.select_dtypes(include=[np.number]).columns.values
    # Fill missing values in the numeric columns.
    df.fillna(train.mean(), inplace=True)
    # check if still any NA's are available in the numerical coulumns of train data
    df[numericCol].isnull().sum().sort_values(ascending=False)
    return df
