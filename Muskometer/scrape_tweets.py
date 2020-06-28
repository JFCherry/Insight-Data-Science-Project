from collections import defaultdict
import os, sys
import time
import pandas as pd
import GetOldTweets3 as got

def scrape_new_tweets(t_last_tweet,username = "elonmusk"):
    """Function to scrape the recent tweets of Elon Musk"""
    #t_last_tweet must be pandas Timestamp data
    os.makedirs('tweet_data', exist_ok=True)
    date_str = str(t_last_tweet.date().year)+"-"\
              +str(t_last_tweet.date().month)+"-"\
              +str(t_last_tweet.date().day)
    count = 0
    # Creation of query object                                                                                                                                                                                      
    tweetCriteria = got.manager.TweetCriteria().setUsername(username)\
                                               .setMaxTweets(count)\
                                               .setSince(date_str)
    # Creation of list that contains all tweets                                                                                                                                                                     
    tweets = None
    for ntries in range(5):
        try:
            tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        except SystemExit:
            print("Trying again in 15 minutes.")
            time.sleep(15*60)
        else:
            break
    if tweets is None:
        print("Failed after 5 tries, quitting!")
        exit(1)

    data = defaultdict(list)
    for t in tweets:
        data["username"].append(username)
        data["tweet_id"].append(t.id)
        data["reply_to"].append(t.to)
        data["date"].append(t.date)
        data["retweets"].append(t.retweets)
        data["favorites"].append(t.favorites)
        data["hashtags"].append(list(set(t.hashtags.split())))
        data["mentions"].append(t.mentions)
        data["text"].append(t.text)
        data["permalink"].append(t.permalink)
    if len(data) == 0: #no new tweets
        return data
    else:
        #make a DataFrame out of the scraped tweets
        df = pd.DataFrame(data, columns=["username","tweet_id",
                                         "reply_to","date","retweets",
                                         "favorites","hashtags","mentions",
                                         "text","permalink"])        
        # Convert 'Time' column to datetime and strip time information.
        df['Time'] = pd.to_datetime(df['date'])
        df.drop(labels=['date'],axis=1,inplace=True)
        return df.sort_values(by='Time',ascending=True)
    
def reload_tweet_data(path,username="elonmusk"):
   #note we'll have to do a .drop and set the 'Time' column to the proper values every time
    df = pd.read_csv(path+username+'_tweets.csv').drop(['Unnamed: 0'],axis='columns')
    #order by earliest first
    df['Time'] = pd.to_datetime(df['Time'])#.sort_values(by='Time',ascending=True)
    return df.sort_values(by='Time',ascending=True).reset_index().drop('index',axis='columns')

def prepend_new_tweets(df_new,df_old): #adds the new tweets to the front of the data set and resets the index
    result = pd.concat([df_old,df_new]).reset_index().drop('index',axis='columns')
    #makes sure we're sorted properly in time order
    result.sort_values(by='Time',ascending=True,inplace=True)
    return result.reset_index().drop('index',axis='columns')

def store_tweet_data(df,path,username="elonmusk"):
    df.to_csv(path+username+'_tweets.csv')
    return
a
def scan_for_new_tweets(path,username="elonmusk"):
    df_old = reload_tweet_data(path,username) #get the old tweets
    #look for new tweets starting from latest date
    df_new = scrape_new_tweets(df_old['Time'].max(),username)
    if df_new == None:# No new tweets
        return df_old
    else:
        df_combined = prepend_new_tweets(df_new,df_old)
        df_combined.drop_duplicates(subset = ['tweet_id'],inplace=True)
        return df_combined

if __name__ == '__main__':
    main()