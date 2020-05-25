# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.core.files.storage import FileSystemStorage
from fbprophet import Prophet
'''from .forecast_models import (
    h2oGBE,
    dataColumns,
    h2oRFE,
    h2oGLE,
    XGBR,
    RFR,
    GBR,
    generateTestDataFrame,
)'''
import csv
import pandas as pd
import numpy as np


def fb_prophet_prediction(df, granular_name, target_column, no_of_days):
    """ Forecasting the timeseries data based on Facebook Prophet."""

    #df = df.rename(columns={'date':'ds', 'Quantity':'y'})
    df = df.rename(columns={'date': 'ds', 'Views': 'y'})
    df["y_orig"] = df['y']
    df['y'] = np.log(df['y'])

    # creating prophet object and training on data
    prophet = Prophet(growth='linear', daily_seasonality=True)
    prophet.fit(df)

    # creating the future dates that prophet will predict
    # specified frequency.
    freq_set = {'Day': 'd', 'Week': 'w', 'Month': 'm'}
    future = prophet.make_future_dataframe(
        freq=freq_set[granular_name], periods=int(no_of_days))

    # predicting the values
    forecast = prophet.predict(future)
    output_df = forecast[['ds', 'yhat']].tail(int(no_of_days))
    npp = np.exp(
        forecast[['yhat', 'yhat_lower', 'yhat_upper']].tail(int(no_of_days)))
    #output_df = forecast[['ds', 'yhat']]
    merged_df = pd.merge(output_df, npp, left_index=True, right_index=True)
    final_output = merged_df[['ds', 'yhat_y']]
    return final_output


def fb_prophet_prediction1(df, granular_name, target_column, no_of_days):
    """ Predict the future dates using FBProphet ML Module"""

    no_of_days = int(no_of_days)
    df = df.rename(columns={'Order Date': 'ds', 'Quantity': 'y'})
    grouped = df.groupby('Sub-Category')

    final = pd.DataFrame()
    for g in grouped.groups:
        group = grouped.get_group(g).sort_values(by='ds')
        m = Prophet(daily_seasonality=True)
        try:
            m.fit(group)
            future = m.make_future_dataframe(periods=no_of_days)
            forecast = m.predict(future)
            forecast = forecast.rename(columns={'yhat': 'yhat_'+g})
            forecast = forecast[['ds', 'yhat_'+g]].tail(no_of_days)
        except:
            # we can use .at(index, col_name) function also
            max_date = group['ds'].iloc()[0]
            list_of_dates_df = pd.date_range(
                start=max_date, periods=no_of_days + 1)
            list_of_dates = [str(pd.to_datetime(i).date())
                             for i in list_of_dates_df.values][1:]
            group = group.loc[group.index.repeat(no_of_days)]
            group['ds'] = list_of_dates
            forecast = group[['ds', 'y']].rename(columns={'y': 'yhat_'+g})
        final = pd.merge(final, forecast.set_index(
            'ds'), how='outer', left_index=True, right_index=True)
    #final = final.reset_index()
    #final.to_csv('Final_op.csv', index=False, encoding='utf-8')
    return final


def create_prediction_dataframe(df, no_of_days):
    """ Create a Prediction DataFrame for the Future Dates"""

    no_of_days = int(no_of_days)
    grouped = df.groupby('Sub-Category')
    list_of_rows = []
    for g in grouped.groups:
        group = grouped.get_group(g).sort_values(by='Order Date')
        max_date = group.iloc[-1][0]
        list_of_dates_df = pd.date_range(
            start=max_date, periods=no_of_days + 1)
        list_of_dates = [str(pd.to_datetime(i).date())
                         for i in list_of_dates_df.values][1:]
        sub_categories = [g]*len(list_of_dates)
        [list_of_rows.append([i, j])
         for i, j in zip(list_of_dates, sub_categories)]
    final_df = pd.DataFrame(list_of_rows, columns=[
                            'Order Date', 'Sub-Category'])
    return final_df

# Create your views here.


@csrf_exempt
def home(request):
    if request.method == "POST":
        algo_name = request.POST.get('algorithm', '')
        granular_name = request.POST.get('granualarity', '')
        target_column = request.POST.get('targetColumn', '')
        data = pd.read_csv(request.FILES['inputGroupSuccess2'])
        '''if not data.empty:
                data['date'] = data['date'].astype('datetime64[ns]')
                data['date'] = data['date'].dt.date'''
        no_of_days = request.POST.get('no_of_days', '')
        # Create Prediction Dataframe
        prediction_df = create_prediction_dataframe(data, no_of_days)
        # generateTestDataFrame(trainData=data, days=no_of_days, gname=granular_name)
        testData = None
        if algo_name == 'XGBRegressor':
            data = XGBR(trainData=data, days=no_of_days, prdData=prediction_df)
        elif algo_name == 'RandomForestRegressor':
            data = RFR(trainData=data, days=no_of_days, prdData=prediction_df)
        elif algo_name == 'GradientBoostingRegressor':
            data = GBR(trainData=data, days=no_of_days, prdData=prediction_df)
        #elif algo_name == 'H2OGradientBoostingEstimator':
        #    data = h2oGBE(trainData=data, testData=testData,
        #                  targetColumn=target_column, featuresColumns=[])
        #elif algo_name == 'H2ORandomForestEstimator':
        #    data = h2oRFE(trainData=data, testData=testData,
        #                  targetColumn=target_column, featuresColumns=[])
        #elif algo_name == 'H2OGeneralizedLinearEstimator':
        #    data = h2oGLE(trainData=data, testData=testData,
        #                  targetColumn=target_column, featuresColumns=[])
        elif algo_name == "FB_Prophet":
            data = fb_prophet_prediction1(
                data, granular_name, target_column, no_of_days)
        data = data.reset_index()
        response = HttpResponse(content_type='application/vnd.csv')
        response['Content-Disposition'] = 'attachment; filename=%s_reult.csv' % (
            algo_name)
        data.to_csv(response, index=False, encoding='utf-8')
        return response
    return render(request, 'test.html')
