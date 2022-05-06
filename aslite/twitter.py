"""
Periodically checks Twitter for tweets about arxiv papers we recognize
and logs the tweets into mongodb database "arxiv", under "tweets" collection.
"""

import os
import re
import time
import math
import pickle
import datetime
import tweepy
import logging

# settings
# -----------------------------------------------------------------------------
sleep_time = 60*10 # in seconds, between twitter API calls. Default rate limit is 180 per 15 minutes

# convenience functions
# -----------------------------------------------------------------------------

def extract_arxiv_pids(r):
  pids = []
  for u in r.get("entities",{}).get("urls",[]):
    m = re.search('arxiv.org/(?:abs|pdf)/(.+)', u.get('unwound_url',u.get('expanded_url','')))
    if m:
      pids.append(m.group(1))
    if m: 
      rawid = m.group(1).strip(".pdf")
      pids.append(rawid)
  return pids

def get_latest_or_loop(q, start_datetime=None):
  if start_datetime is None:
    start_datetime = datetime.datetime.utcnow() - datetime.timedelta(days=6, hours=23)
  start = start_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
  results = []
  next_token = None

  q = "url:arxiv.org lang:en"
  bearer = open('twitter.txt', 'r').read().splitlines()[0]
  client = tweepy.Client(bearer)
  
  while True:
    try:
        resp = client.search_recent_tweets(q, expansions=['author_id'], 
                            max_results=100, 
                            next_token=next_token, 
                            start_time=start, 
                            tweet_fields=['id', 'created_at', 'author_id', 'entities', 'lang', 'public_metrics'],
                            user_fields=['public_metrics'])
        #logging.log(logging.INFO, "fetched %d tweets", len(resp.data))
        results.append(resp)
        next_token = resp.meta.get('next_token', None)
        if next_token is None:
            break
    except Exception as e:
      print('there was some problem (waiting some time and trying again):')
      print(e)
      time.sleep(sleep_time)
  return results

def parse_tweets(results):
    tweets = []
    for result in results:
        authors = result.includes.get('users',[])
        for r in result.data:
            arxiv_pids = extract_arxiv_pids(r)
            if not arxiv_pids: continue # nothing we know about here, lets move on
            author = next(a for a in authors if a.id == r.author_id)

            # create the tweet. intentionally making it flat here without user nesting
            tweet = {}
            tweet['id'] = str(r.id)
            tweet['pids'] = arxiv_pids # arxiv paper ids mentioned in this tweet
            tweet['inserted_at_date'] = datetime.datetime.utcnow().isoformat()
            tweet['created_at_date'] = r.created_at.isoformat()
            tweet['created_at_time'] = int(time.mktime(r.created_at.timetuple()))
            tweet['lang'] = r.lang
            tweet['text'] = r.text
            tweet['retweet_count'] = r.public_metrics.get('retweet_count',0)
            tweet['reply_count'] = r.public_metrics.get('reply_count',0)
            tweet['like_count'] = r.public_metrics.get('like_count',0)
            tweet['quote_count'] = r.public_metrics.get('quote_count',0)
            tweet['user_screen_name'] = author.username
            tweet['user_followers_count'] = author.get('public_metrics',{}).get('followers_count',0)
            tweet['user_following_count'] = author.get('public_metrics',{}).get('following_count',0) 
            tweets.append(tweet)
    return tweets