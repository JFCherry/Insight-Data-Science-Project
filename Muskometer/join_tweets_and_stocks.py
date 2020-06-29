import numpy as np
import pandas as pd
# Make sure inputs are properly formatted as below
#tsla_df = pd.read_csv('../data/raw/tsla_stock_price.csv')\
#                        .drop('Unnamed: 0',axis='columns')
#elon_df = pd.read_csv('../data/raw/elonmusk.csv')\
#                        .drop('Unnamed: 0',axis='columns')
#anomalies_df = pd.read_csv(\
#                        '../data/processed/anomaly_tagged_tweet_features.csv')\
#                        .drop('Unnamed: 0',axis='columns')
#tsla_df['DateTime'] = pd.to_datetime(tsla_df['DateTime'])
#elon_df['Time'] = pd.to_datetime(elon_df['Time'])
#anomalies_df['Time'] = pd.to_datetime(anomalies_df['Time'])
#anomaly_only_df = anomalies_df[anomalies_df['anomalous']!=0.]
#no_anomaly_df = anomalies_df[anomalies_df['anomalous']!=1.]

def nearest(items, pivot): # general get nearest value function
    return min(items, key=lambda x: abs(x - pivot))

def nearest_price(items, pivot, df): # nearest price function
    timestamp = min(items, key=lambda x: abs(x - pivot))
    return df.loc[df['DateTime']==timestamp,['Open']].values[0][0]
    
def join_tweets_and_stocks(stock_df,tweet_df):
    # set new column in the tweet data frame to have the stock date
    tweet_df['stock_time']=tweet_df['Time']\
                         .apply(lambda row: nearest(stock_df['DateTime'],row))
    # set new column in the tweet data frame to have the stock price
    tweet_df['stock_price']=tweet_df['Time']\
                         .apply(lambda row: nearest_price(\
                                            stock_df['DateTime'],row,stock_df))
    return tweet_df

def join_tweets_stocks_anomalies(tweet_df,stock_df,anomalies):
    # set new column in the tweet data frame to have the stock date
    tweet_df['stock_time']=tweet_df['Time']\
                         .apply(lambda row: nearest(stock_df['DateTime'],row))
    # set new column in the tweet data frame to have the stock price
    tweet_df['stock_price']=tweet_df['Time']\
                         .apply(lambda row: nearest_price(\
                                            stock_df['DateTime'],row,stock_df))
    # set the tags for which tweets are anomalous
    tweet_df['anomalous'] = anomalies
    return tweet_df
 
def load_stock_data(stock_name):
    stock_df = pd.read_csv('../data/raw/'+stock_name.lower()+'_stock_price.csv')\
                                .drop('Unnamed: 0',axis='columns')
    stock_df['DateTime'] = pd.to_datetime(stock_df['DateTime'],utc=True)
    return stock_df   
     
def join_tweets_stocks_anomalies_quick(tweet_df,anomalies,stock_name):
    stock_df = load_stock_data(stock_name)
    # send the data we want to arrays
    tweet_array = tweet_df['Time'].values
    time_array = stock_df['DateTime'].values
    stock_time_list = []
    stock_price_list = []
    for i in range(len(tweet_array)):
        bin_now = np.argmin(abs(time_array - tweet_array[i]))
        stock_time_list.append(stock_df.iloc[bin_now]['DateTime'])
        stock_price_list.append(stock_df.iloc[bin_now]['Open'])
    # set new column in the tweet data frame to have the stock date
    tweet_df['stock_time'] = stock_time_list
    # set new column in the tweet data frame to have the stock price
    tweet_df['stock_price'] = stock_price_list
    # set the tags for which tweets are anomalous
    tweet_df['anomalous'] = anomalies
    return tweet_df

if __name__ == '__main__':
    main()