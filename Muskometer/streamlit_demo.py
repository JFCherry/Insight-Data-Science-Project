import streamlit as st
import datetime
import pytz
import numpy as np
import pandas as pd
import time
import altair as alt
from vega_datasets import data

def nearest(items, pivot): #a function to search a set of items
    #and return the closest match
    return min(items, key=lambda x: abs(x - pivot))
#Intro text    
st.title('Muskometer')
st.write('')
st.write("Reading the mind of Elon Musk for fun and profit!  We've tagged \
         some of Elon Musk's tweet which we found strange.  Open up the \
         side bar to see how they vary with sentiment and set up a \
         trading strategy for Tesla stock using Elon's twitter behavior \
         to set buy/sell orders.")
#side bar sliders
poslim = st.sidebar.slider("Positive Sentiment Threshold", 0., 1., .15)
neglim = st.sidebar.slider("Negative Sentiment Threshold", 0., 1., .15)
sell_delay = st.sidebar.slider("Delay for re-buying TSLA", 1, 10, 1)
buy_delay = st.sidebar.slider("Delay for re-selling TSLA", 1, 10, 1)
#convert sider input to Datetime timedelta's
sell_delay = datetime.timedelta(days=sell_delay)
buy_delay = datetime.timedelta(days=buy_delay)
#how much money we have to play with
init_position = st.sidebar.slider(\
                    "Initial position in $", 0., 10_000., 5_000.)
init_capital = st.sidebar.slider(\
                    "Initial capital in $", 0., 10_000., 5_000.)
#Loading datasets
tsla_df = pd.read_csv('../data/raw/tsla_stock_price.csv')\
                        .drop('Unnamed: 0',axis='columns')
elon_df = pd.read_csv('../data/raw/elonmusk.csv')\
                        .drop('Unnamed: 0',axis='columns')
anomalies_df = pd.read_csv(\
            '../data/processed/anomalyandstock_tagged_tweet_features.csv')\
            .drop('Unnamed: 0',axis='columns')
# Fixing the data types from the .csv files
tsla_df['DateTime'] = pd.to_datetime(tsla_df['DateTime'])
elon_df['Time'] = pd.to_datetime(elon_df['Time'])
anomalies_df['Time'] = pd.to_datetime(anomalies_df['Time'])
anomaly_only_df = anomalies_df[anomalies_df['anomalous']!=0.]
no_anomaly_df = anomalies_df[anomalies_df['anomalous']!=1.]

# stock ticker line
line_stock = alt.Chart(tsla_df).mark_line(
    color='black',
    size=3).encode(
    x='DateTime:T',
    y='Open:Q')
#negatively valenced anomalies
points_neganom = alt.Chart(
    anomaly_only_df[anomaly_only_df['text_compound']<-neglim])\
    .mark_point(color='red',size = 200)\
    .encode(x=alt.X('stock_time:T',
            axis=alt.Axis(title='Date')),
            y=alt.Y('stock_price:Q',
            axis=alt.Axis(title='Price')))
#positively valenced anomalies            
points_posanom = alt.Chart(
    anomaly_only_df[anomaly_only_df['text_compound']>poslim])\
    .mark_point(color='#42f542',size = 200)\
    .encode(x=alt.X('stock_time:T',
            axis=alt.Axis(title='Date')),
            y=alt.Y('stock_price:Q',
            axis=alt.Axis(title='Price')))
#neutrally valenced anomalies
points_neuanom = alt.Chart(
    anomaly_only_df[(anomaly_only_df['text_compound']<poslim)\
                    & (anomaly_only_df['text_compound']>-neglim)])\
    .mark_point(color='#ffff26',size = 200)\
    .encode(x=alt.X('stock_time:T',
            axis=alt.Axis(title='Date')),
            y=alt.Y('stock_price:Q',
            axis=alt.Axis(title='Price')))
#ordinary tweets          
points_noanom = alt.Chart(no_anomaly_df)\
    .mark_point(color='steelblue',size = 100)\
    .encode(x=alt.X('stock_time:T',
            axis=alt.Axis(title='Date')),
            y=alt.Y('stock_price:Q',
            axis=alt.Axis(title='Price')))
#puts it all together in one plot
st.altair_chart(points_noanom.interactive() + line_stock.interactive()
                + points_neuanom.interactive()
                + points_posanom.interactive() + points_neganom.interactive()
                ,use_container_width=True)
#sets the starting date for the trading calculation
start_date = st.date_input(label = 'Starting Date', \
                value=datetime.datetime(2015, 1, 1, 0, 0, 0), \
                min_value=datetime.datetime(2010, 6, 3, 0, 0, 0),
                max_value=datetime.datetime.today(), key=None)
# make the dates compatible with pandas
start_date = pd.Timestamp(start_date)
timezone = pytz.timezone('UTC')
start_date = timezone.localize(start_date)
def asset_strategy_calculation(poslim,neglim,init_position,init_capital,
                                buy_delay,sell_delay,start_date,
                                anomaly_only_df,tsla_df):
    """The buying and selling strategy implementing tweet inforation"""
    hold_df = pd.DataFrame() #initialize 
    buy_and_sell_df = pd.DataFrame() #initialize
    #the closest date with a stock price
    true_start_date = nearest(tsla_df['DateTime'],start_date) 
    #the index of true_start_date
    start_index = tsla_df.loc[tsla_df['DateTime'] == true_start_date].index[0]
    hold_df['Time'] = tsla_df['DateTime'].iloc[start_index:] #set dates
    buy_and_sell_df['Time'] = tsla_df['DateTime'].iloc[start_index:] #set dates
    # Position growth scales with Tesla stock price
    hold_df['position'] = (tsla_df['Open'].iloc[start_index:]\
                           /tsla_df['Open'].iloc[start_index])*init_position
    # buy_and sell_df needs to track the number of shares held
    buy_and_sell_df['num_shares'] = init_position/tsla_df['Open']\
                                                    .iloc[start_index]
    # and the value of those shares
    buy_and_sell_df['position'] = (tsla_df['Open'].iloc[start_index:]\
                                   *buy_and_sell_df['num_shares'])
    # Capital growth only changes as a result of buy -> sell orders
    hold_df['capital'] = init_capital
    buy_and_sell_df['capital'] = init_capital
    # Iterate over the anomalies and make trades based on input variables
    for i in anomaly_only_df.index:# we only trade based on tweet anomalies
        #iterate forward through time with the index of anomaly_only_df
        if anomaly_only_df['Time'][i] < start_date: 
            #this anomaly happened before we started trading
            pass
        elif anomaly_only_df['text_compound'][i] < poslim and \
                anomaly_only_df['text_compound'][i] > neglim : 
            #neutral anomaly, do nothing
            pass
        elif anomaly_only_df['text_compound'][i] >= poslim : 
            # buy first then sell
            buy_date = tsla_df.loc[tsla_df['DateTime'] \
                                   == anomaly_only_df['stock_time'][i]]\
                                   ['DateTime']
            buy_index = tsla_df.loc[tsla_df['DateTime'] \
                                    == anomaly_only_df['stock_time'][i]]\
                                    .index[0]
            buy_price = tsla_df.loc[tsla_df['DateTime'] \
                                    == anomaly_only_df['stock_time'][i]]\
                                    ['Open'].values[0]
            sell_date_target = tsla_df.loc[tsla_df['DateTime'] 
                                           == anomaly_only_df['stock_time'][i]\
                                           ]['DateTime'] + buy_delay
            #the desired sell date may not be a business day
            sell_date = nearest(tsla_df['DateTime'].iloc[buy_index+1:],\
                                                    sell_date_target.iloc[0]) 
            #adjust sell_index to buy_and_sell index units with (- start_index)
            sell_index = tsla_df.loc[tsla_df['DateTime'] == sell_date].\
                                                        index[0] - start_index
            sell_price = tsla_df.loc[tsla_df['DateTime'] == sell_date]\
                                                            ['Open'].values[0]
            #the fractional change in our captial from the transaction
            frac_change = sell_price/buy_price 
            buy_and_sell_df['capital'].iloc[sell_index:] *= frac_change
        elif anomaly_only_df['text_compound'][i] <= -neglim : 
            # sell first then buy
            sell_date = tsla_df.loc[tsla_df['DateTime'] 
                       == anomaly_only_df['stock_time'][i]]['DateTime']
            sell_index = tsla_df.loc[tsla_df['DateTime'] 
                                    == anomaly_only_df['stock_time'][i]]\
                                                                .index[0]
            sell_price = tsla_df.loc[tsla_df['DateTime'] \
                                    == anomaly_only_df['stock_time'][i]]\
                                    ['Open'].values[0]
            buy_date_target = tsla_df.loc[tsla_df['DateTime'] \
                                         == anomaly_only_df['stock_time'][i]]\
                                         ['DateTime'] + sell_delay
            #the desired sell date may not be a business day
            buy_date = nearest(tsla_df['DateTime'].iloc[sell_index+1:],\
                               buy_date_target.iloc[0]) 
            #adjust sell_index to buy_and_sell index units with (- start_index)
            buy_index = tsla_df.loc[tsla_df['DateTime'] == buy_date].index[0]\
                        - start_index
            buy_price = tsla_df.loc[tsla_df['DateTime'] == buy_date]['Open']\
                                    .values[0]
            #the change in the number of shares, again adjust sell_index to 
            #buy_and_sell index coords with (- start_index) 
            new_num_shares = sell_price*(buy_and_sell_df['num_shares']\
                                       .iloc[sell_index-start_index])/buy_price 
            #record the new shares
            buy_and_sell_df['num_shares'].iloc[buy_index:] = new_num_shares 
            #compute the new position
            buy_and_sell_df['position'].iloc[buy_index:] = new_num_shares*\
                                tsla_df['Open'].iloc[buy_index+start_index:]
            
    hold_df['total'] = hold_df['position']+hold_df['capital']
    buy_and_sell_df['total'] = buy_and_sell_df['position']+\
                               buy_and_sell_df['capital']
    #relative performance
    hold_df['relative'] = hold_df['total']/hold_df['total'] - 1.
    buy_and_sell_df['relative'] = buy_and_sell_df['total']/\
                                  hold_df['total'] - 1.
    return buy_and_sell_df,hold_df    
#run a simulation to evaluate our buy and sell strategy
strat_df,hold_df = asset_strategy_calculation(poslim,neglim,init_position,\
                                init_capital,buy_delay,sell_delay,start_date,\
                                anomaly_only_df,tsla_df)
#Plot it all up!
st.write("Your projected portfolio performance over time:")
# stock ticker line
line_hold = alt.Chart(hold_df).mark_line(
    color='black',
    size=2).encode(x=alt.X('Time:T',
    axis=alt.Axis(title='Date')),
    y=alt.Y('total:Q',
    axis=alt.Axis(title='Portfolio Value ($)')))
# strategy ticker line
line_bs = alt.Chart(strat_df).mark_line(
    color='red',
    size=3).encode(x=alt.X('Time:T',
    axis=alt.Axis(title='Date')),
    y=alt.Y('total:Q',
    axis=alt.Axis(title='Portfolio Value ($)')))
# plot it all up
st.altair_chart(line_hold.interactive() + line_bs.interactive()
                ,use_container_width=True)
# Do a relative performance plot
# null performance tracker line
line_null = alt.Chart(hold_df).mark_line(
    color='black',
    size=1).encode(x=alt.X('Time:T',
    axis=alt.Axis(title='Date')),
    y=alt.Y('relative:Q',
    axis=alt.Axis(title='Relative Performance')))
# relative performance tracker line
line_model = alt.Chart(strat_df).mark_line(
    color='green',
    size=3).encode(x=alt.X('Time:T',
    axis=alt.Axis(title='Date')),
    y=alt.Y('relative:Q',
    axis=alt.Axis(title='Relative Performance')))
# plot it all up
st.altair_chart(line_null.interactive() + line_model.interactive()
                ,use_container_width=True)


####### Vega stuff for a demo
#source = data.seattle_weather()
#source.dtypes
#source
#line = alt.Chart(source).mark_line(
#    color='red',
#    size=3
#).transform_window(
#    rolling_mean='mean(temp_max)',
#    frame=[-15, 15]
#).encode(
#    x='date:T',
#    y='rolling_mean:Q'
#)
#
#points = alt.Chart(source).mark_point().encode(
#    x='date:T',
#    y=alt.Y('temp_max:Q',
#            axis=alt.Axis(title='Max Temp'))
#)
#
#points.interactive() + line.interactive()
###################
## Generate some random data
#df = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
#
## Build a scatter chart using altair. I modified the example at
## https://altair-viz.github.io/gallery/scatter_tooltips.html
#scatter_chart = st.altair_chart(
#    alt.Chart(df)
#        .mark_circle(size=60)
#        .encode(x='a', y='b', color='c')
#        .interactive()
#)
#
## Append more random data to the chart using add_rows
#for ii in range(0, 100):
#    df = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
#    scatter_chart.add_rows(df)
#    # Sleep for a moment just for demonstration purposes, so that the new data
#    # animates in.
#    time.sleep(0.1)