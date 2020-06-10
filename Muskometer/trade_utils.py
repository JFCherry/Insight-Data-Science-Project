import datetime
import pytz
import numpy as np
import pandas as pd
import time
import random

def apply_rules(index_i,index_j,stock_np,anomaly_np,buy_sell_np,rule,
                buy_delay,sell_delay,start_index,rand_flag = False):
    #This is only for modelling random guessing########
    if rand_flag == True:
        rule = random.choice(['buy','nothing','sell'])
    else:
        pass
    ###################################################
    if rule == 'nothing':
        return buy_sell_np
    elif rule == 'buy':
        # buy first then sell
        buy_date = anomaly_np[index_j,2]
        buy_index = np.argmin(np.abs(stock_np[:,0] - anomaly_np[index_j,2]))
        buy_price = stock_np[buy_index,1]
        sell_date_target = buy_date + buy_delay
        #the desired sell date may not be a business day
        diff = stock_np[:,0] - sell_date_target
        mask = np.ma.less_equal(diff, 0)
        #this will be the index of the true sell date
        sell_index = np.argmin(np.abs(mask))
        if sell_index == 0: #we are off the edge of the map
            #the sell date is after the end of the stock data
            #do nothing and return
            return buy_sell_np
        sell_price = stock_np[sell_index,1]
        #the fractional change in our captial from the transaction
        frac_change = sell_price/buy_price 
        #shift the total capital for all days after the transaction
        #-start_index corrects for offset in indices with stock_np
        buy_sell_np[sell_index - start_index:,3] *= frac_change
        return buy_sell_np
    elif rule == 'sell':
        sell_date = anomaly_np[index_j,2]
        sell_index = np.argmin(np.abs(stock_np[:,0] - anomaly_np[index_j,2]))
        sell_price = stock_np[sell_index,1]
        buy_date_target = sell_date + sell_delay
        #the desired buy date may not be a business day
        diff = stock_np[:,0] - buy_date_target
        mask = np.ma.less_equal(diff, 0)
        #this will be the index of the true buy date
        buy_index = np.argmin(np.abs(mask))
        if buy_index == 0: #we are off the edge of the map
            #the buy date is after the end of the stock data
            #do nothing and return
            return buy_sell_np
        buy_price = stock_np[buy_index,1]
        #the change in the number of shares, again adjust sell_index to 
        #buy_and_sell index coords with (- start_index) 
        new_num_shares = sell_price*(buy_sell_np[sell_index-start_index,1])/buy_price 
        #record the new shares
        #-start_index corrects for offset in indices with stock_np
        buy_sell_np[buy_index - start_index:,1] = new_num_shares 
        #compute the new position
        #-start_index corrects for offset in indices with stock_np
        buy_sell_np[buy_index - start_index:,2] = new_num_shares*stock_np[buy_index:,1]
        return buy_sell_np
    else: #something went wrong
        return buy_sell_np

def convert_dataframes_to_numpy(start_date,stock_df,anomaly_only_df,
                                buy_delay,sell_delay):
    """Converts dataframes and delta times to numpy arrays and floats."""
    #convert datetimes to floats (s after start date)
    tsla_time = convert_to_seconds_after_start(start_date,stock_df,'DateTime')
    anomaly_time = convert_to_seconds_after_start(start_date,anomaly_only_df,'Time')
    anomaly_stock_time = convert_to_seconds_after_start(start_date,anomaly_only_df,'stock_time')
    #strip inputs down to only the necessary entries for numpy array assignment
    tsla_np = stock_df[['DateTime','Open']].values
    anomaly_only_np = anomaly_only_df[['Time','stock_time',\
                                       'text_compound']].values
    #'index' goes on index 0 
    anomaly_only_np = np.vstack((anomaly_only_df.index,\
                                 anomaly_only_np.transpose())).transpose()
    # write time in (s) to the arrays
    tsla_np[:,0] = tsla_time
    anomaly_only_np[:,1] = anomaly_time
    anomaly_only_np[:,2] = anomaly_stock_time
    # Convert the sell_delay and buy_delay to seconds
    sell_delay = sell_delay.total_seconds()
    buy_delay = buy_delay.total_seconds()
    start_time = 0.
    return tsla_np,anomaly_only_np,sell_delay,buy_delay,start_time

def convert_to_seconds_after_start(start_time,df,time_column):
    """A function to turn the datetime data from the
        pandas data frames to floats for the numpy
        vertion of the asset strategy model."""
    df['temporary'] = df.apply(lambda row : (row[time_column] - start_time).total_seconds(),axis=1)
    #returns a numpy array
    result = np.zeros(df.shape[0])
    result = df['temporary'].values
    df.drop(columns=['temporary'])
    return result

def convert_trading_to_df(df,start_date):
    out_df = pd.DataFrame(data=df,    # values
                     index=range(len(df[:,0])),    # 1st column as index
                     columns=['time_in_sec','num_shares','position',\
                             'capital','total','relative'])
    date_list = [start_date + \
                     datetime.timedelta(seconds = sec) for sec in df[:,0]]
    out_df['Time'] = date_list
    #datetime.timedelta(days=1)
    return out_df

def asset_strategy_calculation_numpy(poslim,neglim,init_position,init_capital,\
                                    buy_delay,sell_delay,anomaly_only_np,\
                                    tsla_np,start_time,rule_pos,\
                                    rule_neu,rule_neg,rand_flag = False):
    """The buying and selling strategy implementing tweet inforation""" 
    #the index of true_start_date
    start_index = np.argmin(np.abs(tsla_np[:,0]-start_time)) 
    hold_np = np.zeros([len(tsla_np[start_index:,0]),6]) #initialize 
    buy_and_sell_np = np.zeros([len(tsla_np[start_index:,0]),6]) #initialize
    hold_np[:,0] = tsla_np[start_index:,0] #set dates
    buy_and_sell_np[:,0] = tsla_np[start_index:,0] #set dates
    # Position growth scales with Tesla stock price
    hold_np[:,2] = (tsla_np[start_index:,1]\
                           /tsla_np[start_index,1])*init_position
    # hold_np does not need to track the number of shares held
    hold_np[:,1] = init_position/tsla_np[start_index,1]
    # buy_and sell_np needs to track the number of shares held
    buy_and_sell_np[:,1] = init_position/tsla_np[start_index,1]
    # and the value of those shares
    buy_and_sell_np[:,2] = (tsla_np[start_index,1]*buy_and_sell_np[:,1])
    # Capital growth only changes as a result of buy -> sell orders
    hold_np[:,3] = init_capital
    buy_and_sell_np[:,3] = init_capital
    # Iterate over the anomalies and make trades based on input variables
    j = 0 #iteration variable for anomaly_only_np
    # i is the index of the stock_np time
    for i in anomaly_only_np[:,0]:# we only trade based on tweet anomalies
        #iterate forward through time with the index of anomaly_only_df
        if anomaly_only_np[j,2] < start_time: 
            #this anomaly happened before we started trading
            #do nothing
            pass
        elif anomaly_only_np[j,3] < poslim and \
                anomaly_only_np[j,3] > -neglim : #we have a neutral anomaly
            #apply defined trading rule
            buy_and_sell_np = apply_rules(i,j,tsla_np,anomaly_only_np,\
                                          buy_and_sell_np,rule_neu,\
                                          buy_delay,sell_delay,\
                                          start_index,rand_flag)
            pass
        elif anomaly_only_np[j,3] >= poslim : #we have a positive anomaly
            #apply defined trading rule
            buy_and_sell_np = apply_rules(i,j,tsla_np,anomaly_only_np,\
                                          buy_and_sell_np,rule_pos,\
                                          buy_delay,sell_delay,\
                                          start_index,rand_flag)
            
        elif anomaly_only_np[j,3] <= -neglim : 
            # apply defined trading rule
            buy_and_sell_np = apply_rules(i,j,tsla_np,anomaly_only_np,\
                                          buy_and_sell_np,rule_neg,\
                                          buy_delay,sell_delay,\
                                          start_index,rand_flag)
        j += 1 #increment the anomaly index
            
            
    hold_np[:,4] = hold_np[:,2]+hold_np[:,3]
    buy_and_sell_np[:,4] = buy_and_sell_np[:,2]+buy_and_sell_np[:,3]
    #relative performance
    hold_np[:,5] = hold_np[:,4]/hold_np[:,4]
    buy_and_sell_np[:,5] = buy_and_sell_np[:,4]/hold_np[:,4]
    return buy_and_sell_np,hold_np
    
def grid_search_trading_algo(stock_df,anomaly_only_df,start_date):
    #parameters
    timezone = pytz.timezone('UTC')
    start_date = timezone.localize(datetime.datetime(2015,1,1))
    pos_lims = np.linspace(0.,1.,11)
    neg_lims = np.linspace(0.,1.,11)
    buy_delays = np.linspace(1,10,10)*86400. # delay time in seconds
    sell_delays = np.linspace(1,10,10)*86400. # delay time in seconds
    pos_rules = ['buy','nothing','sell']
    neu_rules = ['buy','nothing','sell']
    neg_rules = ['buy','nothing','sell']
    #convert dataframes to numpy input
    tsla_np,anomaly_only_np,sell_delay,buy_delay,start_time = \
            convert_dataframes_to_numpy(start_date,stock_df,anomaly_only_df,
                                        datetime.timedelta(days=1),datetime.timedelta(days=1))
    #array for output
    #use fractional performance as the test metric
    output = np.zeros([11,11,10,10,3,3,3],np.double)
    for i in range(len(pos_lims)):
        for j in range(len(neg_lims)):
            for k in range(len(buy_delays)):
                for l in range(len(sell_delays)):
                    for m in range(len(pos_rules)):
                        for n in range(len(neu_rules)):
                            for o in range(len(neg_rules)):
                                #print (i,j,k,l,m,n,o)
                                #print (pl,nl,buy_d,sell_d,posr,neur,negr,start_time)
                                temp1,temp2 = asset_strategy_calculation_numpy\
                                                        (pos_lims[i],neg_lims[j],5000.,5000.,\
                                                        buy_delays[k],sell_delays[l],anomaly_only_np,\
                                                        tsla_np,start_time,pos_rules[m],\
                                                        neu_rules[n],neg_rules[o])
                                output[i,j,k,l,m,n,o] = temp1[-1,5] #this is the final fractional performance
                                
    return output
    
def test_random_tweets(stock_df,anomalies_df):
    #from previous results: 
    timezone = pytz.timezone('UTC')
    start_date = timezone.localize(datetime.datetime(2015,1,1))
    pos_lims = 0.
    neg_lims = .8
    buy_delays = 10.*86400
    sell_delays = 1.*86400
    pos_rules = 'buy'
    neu_rules = 'buy'
    neg_rules = 'sell'
    #create initial array to old the output of each run
    final_results = np.array([])
    for i in range(1000):#do one thousand samples
        #make fake list of anomalies from real tweets
        #using the real sentiment values from each
        anomaly_only_df = anomalies_df.sample(98).sort_values(by=['stock_time'])
        #convert dataframes to numpy input
        tsla_np,anomaly_only_np,dummy1,dummy2,start_time = \
                convert_dataframes_to_numpy(start_date,stock_df,anomaly_only_df,
                                            datetime.timedelta(days=1),datetime.timedelta(days=1))
        #uncomment these lines for true random buy/sell orders
        #run1,temp1 = asset_strategy_calculation_numpy\
        #                                    (pos_lims,neg_lims,5000.,5000.,\
        #                                    buy_delays,sell_delays,anomaly_only_np,\
        #                                    tsla_np,start_time,pos_rules,\
        #                                    neu_rules,neg_rules,rand_flag = True)
        run1,temp1 = asset_strategy_calculation_numpy\
                                            (pos_lims,neg_lims,5000.,5000.,\
                                            buy_delays,sell_delays,anomaly_only_np,\
                                            tsla_np,start_time,pos_rules,\
                                            neu_rules,neg_rules)
        if final_results.shape == (0,):#if this is the first run through
            final_results = run1.reshape(1,run1.shape[0],run1.shape[1])
        else:#stack the results into a single numpy array
            final_results = np.vstack((run1.reshape(1,run1.shape[0],run1.shape[1]),final_results))
    return final_results