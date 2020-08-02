import nltk
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
import re


def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"RT", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),:;!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    df = df[df[text_field].notna()]
    return df

def apply_vader(tweet,category,sid): #gives back the float value of the vader sentiment
    return sid.polarity_scores(tweet)[category]

def tokenize_hashtags(df): #tokenize hashtags, using one hot encoding
    df['hashtags_token'] = 0. #initialize all to zero
    df['hashtags_token'].loc[df['hashtags'] != '[]'] = 1. #any field with a hashtag set to 1.
    return df

def tokenize_mentions(df): #tokenize mentions, using one hot encoding
    df['mentions_token'] = 0. #initialize all to zero
    df['mentions_token'].loc[df['mentions'].notna()] = 1. #any field with a mention set to 1.
    return df

def tokenize_reply_to(df): #tokenize mentions, using one hot encoding
    df['reply_to_token'] = 0. #initialize all to zero
    df['reply_to_token'].loc[df['reply_to'].notna()] = 1. #any field with a reply_to set to 1.
    return df

def convert_hashtag(input_txt):
    if input_txt == '[]': #return empty string if no hashtag
        return ""
    input_list = input_txt.strip("['']").split("', '") #strips out useless characters
    txt_list = re.findall('[A-Z][^A-Z]*', " ".join(input_list)) #splits hastags into words on Captial letters
    return " ".join(txt_list)

def tweet_word_count(df):
    tokenizer = RegexpTokenizer(r'\w+') #split on words
    df["tokens"] = df["tweet"].apply(tokenizer.tokenize) #returns list of individual words
    df['tweet_length'] = df.apply(lambda row : len(row['tokens']), axis=1) #creates tweet length column
    df = df.drop(['tokens'],axis='columns') #drops the temporary column
    return df   

def integral_history(df,category,length):
    #the depth back in tweet history
    result = df[category]
    
    
def construct_features(tweets):
    """Constructs features from Elon's tweet data"""
    #generate the sentiment intensity analyzer instance
    try:
        sid = SentimentIntensityAnalyzer() #returns error if no lexicon
    except:
        nltk.download('vader_lexicon') #get the bloody lexicon
        sid = SentimentIntensityAnalyzer() #returns error if no lexicon
    # Clean the text of the tweets
    tweets = standardize_text(tweets,"text")
    # Tokenize the hashtags
    tweets = tokenize_hashtags(tweets)
    # Tokenize the mentions
    tweets = tokenize_mentions(tweets)
    # Tokenize the reply_to
    tweets = tokenize_reply_to(tweets)
    # Clean the text of the hastags
    tweets["hashtags"] = tweets.apply(lambda row: convert_hashtag(row['hashtags']),axis=1)
    # Prepare to apply vader to the tweets
    vader_categories = ['neg','neu','pos','compound']
    # Apply vader to the tweets
    for cat in vader_categories: #iterates over the categories
        #creates new feature each iteration
        tweets['text_'+cat] = tweets.apply(lambda row : apply_vader(row['text'],cat,sid), axis=1)
    # Apply vader to the hashtags
    for cat in vader_categories: #iterates over the categories
        #creates new feature each iteration
        tweets['hashtags_'+cat] = tweets.apply(lambda row : apply_vader(row['hashtags'],cat,sid), axis=1)
    #Do some temporal processing
    #Hour of the day
    tweets['hour'] = tweets['Time'].dt.hour
    #Time between tweets in seconds
    tweets['delta_time'] = abs(pd.to_timedelta((tweets['Time']-tweets['Time']\
                                                     .shift()).fillna(6000.)).astype('timedelta64[s]'))\
                                                     .replace(0.,6000.)
    tweets['log10_delta_time'] = np.log10(abs(pd.to_timedelta((tweets['Time']-tweets['Time']\
                                                     .shift()).fillna(60.)).astype('timedelta64[s]')\
                                                     .replace(0.,6000.)))
    #Make some rate of sentiment change features
    tweets['dcompound_dTime'] = (tweets['text_compound']-tweets['text_compound']
                                           .shift()).fillna(0.)/(tweets['delta_time']) #change per second
    tweets['dcompound_dTweet'] = (tweets['text_compound']-tweets['text_compound']
                                            .shift()).fillna(0.) #change per tweet
    #Make some integral sentiment change features
    tweets['integral_compound_5'] = tweets['text_compound'].rolling(min_periods=1, window=5).sum()
    tweets['integral_compound_10'] = tweets['text_compound'].rolling(min_periods=1, window=10).sum()
    #Make a difference sentiment features
    tweets['delta_compound_mean'] = tweets['text_compound'] - tweets['text_compound'].mean()
    tweets['delta_compound_median'] = tweets['text_compound'] - tweets['text_compound'].median()
    #All done for now
    return tweets

def strip_down_to_features_and_rescale(df):
    #drop improperly formatted data
    df = df.drop(['username','reply_to','retweets',
                  'tweet_id','favorites','hashtags','mentions',
                  'text','permalink','Time'],axis='columns')
    # These features are on a 0 to 1 scale
    zero_to_one = ['hour','delta_time','log10_delta_time']
    # These features are on a -1 to 1 scale
    negone_to_one = ['dcompound_dTime','dcompound_dTweet','integral_compound_5',
                    'integral_compound_10','delta_compound_mean','delta_compound_median']
    # shrink the scale for the zero_to_one features
    df[zero_to_one] /= df[zero_to_one].max()
    # shrink the scale for the -1 to 1 ranges
    # need to preserve true zero, however, so no shifting the mean
    for x in negone_to_one:
        # this won't fill the entire range -1 to 1, but it preserves true 0
        df[x] /= max(abs(df[x].min()),df[x].max())
    return df

def combine_with_old_unscaled_tweet_features_and_store(df,username = 'elonmusk'):
    """Function to take the new tweets +10 data frame, process it for features,
        and combine it with the old unscaled features data."""
    # load old unscaled tweet data
    uf_oldtweets_df = pd.read_csv('../data/cleaned/'+username+'_unscaled_tweet_features.csv')\
                                .drop('Unnamed: 0',axis='columns')
    uf_oldtweets_df['Time'] = pd.to_datetime(uf_oldtweets_df['Time'])
    # compute the new tweet features
    uf_newtweets_df = construct_features(df)
    # combine the two data frames
    result = pd.concat([uf_oldtweets_df,uf_new_tweets_df]).reset_index().drop('index',axis='columns')
    # makes sure we're sorted properly in time order
    result.sort_values(by='Time',ascending=True,inplace=True)
    # eliminates duplicate entries
    result.drop_duplicates(subset = ['Time'],inplace = True)
    # store the results
    result.to_csv('../data/cleaned/'+username+'_unscaled_tweet_features.csv')
    return result

if __name__ == '__main__':
    main()