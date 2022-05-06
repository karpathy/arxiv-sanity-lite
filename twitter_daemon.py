"""
This script is intended to wake up every 30 min or so (eg via cron),
it checks for any new arxiv papers via the arxiv API and stashes
them into a sqlite database.
"""

import sys
import time
import random
import datetime
import logging
import argparse

from aslite.twitter import get_latest_or_loop, parse_tweets
from aslite.db import get_papers_db, get_tweets_db

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    pdb = get_papers_db(flag='c')
    tdb = get_tweets_db(flag='c')
    prevn = len(tdb)

    def store(t):
        tdb[t['id']] = t
    
    latest = 0
    for k,v in tdb.items():
        if v['created_at_time'] > latest:
            latest = v['created_at_time']

    if prevn > 0:
        start = datetime.datetime.utcfromtimestamp(latest)
    else: 
        start = None

    # fetch the latest tweets mentioning arxiv.org
    results = get_latest_or_loop(start)
    tweets = parse_tweets(results)
    for t in tweets:
        if t['id'] not in tdb: store(t)
