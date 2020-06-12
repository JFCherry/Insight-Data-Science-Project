# -*- coding: utf-8 -*-
import streamlit as st
import datetime
import pytz
import numpy as np
import pandas as pd
import time
import trade_utils as tu
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#Intro text    
st.title('Muskometer')
st.write('')
st.markdown("### Reading the mind of Elon Musk for fun and profit!  When Elon \
         tweets something unusual, the market notices.  We've used a neural \
         network to identify \
         some of Elon Musk's tweets which are out of character.  Open up the \
         side bar to see how those strange tweets vary with sentiment \
         and set up a \
         trading strategy for Tesla stock using Elon's twitter behavior \
         to set buy/sell orders.")

#side bar sliders
poslim = st.sidebar.slider("Positive Sentiment Threshold (buy trigger)",\
                                                                0., 1., .6)
neglim = st.sidebar.slider("Negative Sentiment Threshold (sell trigger)",\
                                                                0., 1., .3)
rule_pos = st.sidebar.selectbox(\
                    "What type of trade is executed on a Positive anomaly?",\
                    ("buy","nothing","sell"), index = 0)
rule_neu = st.sidebar.selectbox(\
                    "What type of trade is executed on a Neutral anomaly?",\
                    ("buy","nothing","sell"), index = 2)
rule_neg = st.sidebar.selectbox(\
                    "What type of trade is executed on a Negative anomaly?",\
                    ("buy","nothing","sell"), index = 0)
sell_delay = 1.#st.sidebar.slider("Delay for re-buying TSLA (after sell order)",\
               #                                                 1, 10, 1)
buy_delay = 1.#st.sidebar.slider("Delay for re-selling TSLA (after buy order)",\
               #                                                 1, 10, 10)
#convert sider input to Datetime timedelta's
sell_delay = datetime.timedelta(days=sell_delay)
buy_delay = datetime.timedelta(days=buy_delay)
#how much money we have to play with
init_position = st.sidebar.slider(\
                                    "Initial position in $ (how much stock)",\
                                    0., 10_000., 5_000.)
init_capital = st.sidebar.slider(\
                                    "Initial capital in $ (how much cash",\
                                    0., 10_000., 5_000.)
#Loading datasets
tsla_df = pd.read_csv('../data/raw/tsla_stock_price.csv')\
                        .drop('Unnamed: 0',axis='columns')
elon_df = pd.read_csv('../data/raw/elonmusk_tweets.csv')\
                        .drop('Unnamed: 0',axis='columns')\
                        .sort_values(by = 'Time')
anomalies_df = pd.read_csv(\
            '../data/processed/elonmusk_anomalyandstock_tagged_tweet_features.csv')\
            .drop('Unnamed: 0',axis='columns')
elon_df['Time'] = pd.to_datetime(elon_df['Time'])
anomalies_df['Time'] = pd.to_datetime(anomalies_df['Time'])
anomalies_df['stock_time'] = pd.to_datetime(anomalies_df['stock_time'])
anomaly_only_df = anomalies_df[anomalies_df['anomalous']!=0.]
no_anomaly_df = anomalies_df[anomalies_df['anomalous']!=1.]
tsla_df['DateTime'] = pd.to_datetime(tsla_df['DateTime'])
# anomaly tagged and sentiment tagged tweets
neg_comp = anomaly_only_df[anomaly_only_df['text_compound']<-neglim]
neu_comp = anomaly_only_df[(anomaly_only_df['text_compound']<poslim)\
                    & (anomaly_only_df['text_compound']>-neglim)]
pos_comp = anomaly_only_df[anomaly_only_df['text_compound']>poslim]
# Put the original 


fig1 = go.Figure()

#fig1.add_trace(go.Scatter(x = no_anomaly_df['stock_time'],
#                          y = no_anomaly_df['stock_price'],
#                          mode = 'markers', name = 'Normal Tweets',
#                          hoverinfo='skip',
#                          marker = dict(color='rgba(70, 130, 180, .1)',\
#                          size=10, \
#                          line=dict(color='rgba(70, 130, 180, 1.)', width=2)
#               )
#    )
#)



fig1.add_trace(go.Scatter(x=tsla_df['DateTime'],
                          y=tsla_df['Open'],
                          line=dict(color = 'rgb(0,0,0)',width=2),
                          mode = 'lines', name='Tesla stock price'
    )
)




fig1.add_trace(go.Scatter(x = neu_comp['stock_time'],\
               y = neu_comp['stock_price'],
               mode = 'markers',name = 'Neutral Anomalies',
               hovertemplate = '%{text}',text = [txt for txt in \
                                                neu_comp['text'].values],
               marker = dict(color='rgba(255,255,0,.1)',\
               size=15, line=dict(color='rgba(255,255,0,.8)',width=2)
               )
    )
)

fig1.add_trace(go.Scatter(x = pos_comp['stock_time'],\
               y = pos_comp['stock_price'],
               mode = 'markers',name = 'Positive Anomalies',
               hovertemplate = '%{text}',text = [txt for txt in \
                                                pos_comp['text'].values],
               marker = dict(color='rgba(0,255,0,.1)',\
               size=15, line=dict(color='rgba(0,255,0,.8)',width=2)
               )
    )
)

fig1.add_trace(go.Scatter(x = neg_comp['stock_time'],\
               y = neg_comp['stock_price'],
               mode = 'markers',name = 'Negative Anomalies',
               hovertemplate = '%{text}',text = [txt for txt in \
                                                neg_comp['text'].values],
               marker = dict(color='rgba(255,0,0,.1)',\
               size=15, line=dict(color='rgba(255,0,0,.8)',width=2)
               )
    )
)


fig1.update_layout(
    title="Tweet Anomalies and Stock Price",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    width=800, height=400,
    plot_bgcolor = 'rgba(67,70,75,0.1)',
    font=dict(family="IBM Plex Sans",
        size=18,
        color="#262730"
    )
)

fig1.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black',
                  showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,.4)')
fig1.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black',
                  showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,.4)')

st.plotly_chart(fig1,use_container_width=False) 
#sets the starting date for the trading calculation
start_date = st.date_input(label = 'Start trading on which day?', \
                value=datetime.datetime(2016, 1, 1, 0, 0, 0), \
                min_value=datetime.datetime(2010, 6, 3, 0, 0, 0),
                max_value=datetime.datetime.today(), key=None)
#sets the ending date for the trading calculation
end_date = st.date_input(label = 'End trading on which day?', \
                value=datetime.datetime(2020, 2, 1, 0, 0, 0), \
                min_value=datetime.datetime(2010, 6, 3, 0, 0, 0),
                max_value=datetime.datetime.today(), key=None)
# make the dates compatible with pandas
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)
timezone = pytz.timezone('UTC')
start_date = timezone.localize(start_date)
end_date = timezone.localize(end_date)
# Convert data frames to numpy format for faster processing 
tsla_np,anomaly_only_np,sell_delay,buy_delay,start_time,end_time = \
            tu.convert_dataframes_to_numpy(start_date, end_date,tsla_df,\
                                        anomaly_only_df, buy_delay,sell_delay)
# Run the asset strategy calculation
strat_np,hold_np = tu.asset_strategy_calculation_numpy(\
                                    poslim,neglim,init_position,init_capital,\
                                    buy_delay,sell_delay,anomaly_only_np,\
                                    tsla_np,start_time,end_time,rule_pos,\
                                    rule_neu,rule_neg)
# Convert the numpy arrays back to data frames
strat_df = tu.convert_trading_to_df(strat_np,start_date) 
hold_df = tu.convert_trading_to_df(hold_np,start_date)                         
# plot up the return data                         
fig2 = make_subplots(specs=[[{"secondary_y": True}]]) 
fig2.add_trace(go.Scatter(x=hold_df['Time'],
                          y=hold_df['total'],
                          line=dict(color = 'rgb(0,0,0)',width=2),
                          mode = 'lines', name='Held Tesla Stock'
    )
)
fig2.add_trace(go.Scatter(x=strat_df['Time'],
                          y=strat_df['total'],
                          line=dict(color = 'rgba(203,65,84,1.)',width=2),
                          mode = 'lines', name='Traded Tesla Stock'
    )
)
#fig2.add_trace(go.Scatter(x=strat_df['Time'],
#                          y=strat_df['total']/(init_position+init_capital),
#                          line=dict(color = 'rgba(203,65,84,1.)',width=2),
#                          mode = 'lines', name='Traded Tesla Stock'
#    ),secondary_y=True
#)
fig2.update_layout(
    title="Portfolio Value",
    xaxis_title="Date",
    yaxis_title="Total Value ($)",
    width=800, height=400,
    plot_bgcolor = 'rgba(67,70,75,0.1)',
    font=dict(family="IBM Plex Sans",
        size=18,
        color="#262730"
    )
)

fig2.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black',
                  showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,.4)')
fig2.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black',
                  showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,.4)')
#fig2.update_yaxes(title_text="ROI",secondary_y=True)  
st.plotly_chart(fig2,use_container_width=False) 
# Make a Return on Investment plot
#fig3 = go.Figure() 
#fig3.add_trace(go.Scatter(x=hold_df['Time'],
#                          y=hold_df['total']/(init_position+init_capital),
#                          line=dict(color = 'rgb(0,0,0)',width=2),
#                          mode = 'lines', name='Held Tesla Stock'
#    )
#)
#fig3.add_trace(go.Scatter(x=strat_df['Time'],
#                          y=strat_df['total']/(init_position+init_capital),
#                          line=dict(color = 'rgba(50, 168, 82, 1.)',width=2),
#                          mode = 'lines', name='Traded Tesla Stock'
#    )
#)
#fig3.update_layout(
#    title="Return on Investment",
#    xaxis_title="Date",
#    yaxis_title="Final Value / Initial Value",
#    width=800, height=400,
#    plot_bgcolor = 'rgba(67,70,75,0.1)',
#    font=dict(family="IBM Plex Sans",
#        size=18,
#        color="#262730"
#    )
#)
# Make a Relative Performance
fig3 = go.Figure() 
fig3.add_trace(go.Scatter(x=hold_df['Time'],
                          y=hold_df['relative'],
                          line=dict(color = 'rgb(0,0,0)',width=2),
                          mode = 'lines', name='Held Tesla Stock'
    )
)
fig3.add_trace(go.Scatter(x=strat_df['Time'],
                          y=strat_df['relative'],
                          line=dict(color = 'rgba(50, 168, 82, 1.)',width=2),
                          mode = 'lines', name='Traded Tesla Stock'
    )
)
fig3.update_layout(
    title="Relative Performance",
    xaxis_title="Date",
    yaxis_title="Traded Tesla / Held Tesla",
    width=800, height=400,
    plot_bgcolor = 'rgba(67,70,75,0.1)',
    font=dict(family="IBM Plex Sans",
        size=18,
        color="#262730"
    )
)



fig3.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black',
                  showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,.4)')
fig3.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black',
                  showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,.4)') 
st.plotly_chart(fig3,use_container_width=False) 

# Display our prediction for Elon's next tweet
valence_str = "Positive"
anomaly_str = "Normal"
st.title("Elon's next tweet is likely to be:")
st.markdown("## "+valence_str+" and "+anomaly_str+".")
# Ask the user to sign up for email alerts
st.title("Email notifications:")
st.markdown("### Save your preferred trading model and sign up to be\
            notified with instructions the next time Elon is likely to\
            tweet something out of the ordinary.")
user_model = [poslim,neglim,rule_pos,rule_neu,rule_neg,buy_delay,sell_delay]
user_email = st.text_input("email:", "")