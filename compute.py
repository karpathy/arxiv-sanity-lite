"""
Extracts tfidf features from all paper abstracts and saves them to disk.
"""

import pickle
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from aslite.db import get_papers_db, save_features

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arxiv Computor')
    parser.add_argument('-n', '--num', type=int, default=20000, help='number of tfidf features')
    parser.add_argument('--min_df', type=int, default=5, help='min df')
    parser.add_argument('--max_df', type=float, default=0.1, help='max df')
    args = parser.parse_args()
    print(args)

    v = TfidfVectorizer(input='content',
                        encoding='utf-8', decode_error='replace', strip_accents='unicode',
                        lowercase=True, analyzer='word', stop_words='english',
                        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
                        ngram_range=(1, 2), max_features=args.num,
                        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                        max_df=args.max_df, min_df=args.min_df)

    pdb = get_papers_db(flag='r')

    def make_corpus():
        for p, d in pdb.items():
            author_str = ' '.join([a['name'] for a in d['authors']])
            yield ' '.join([d['title'], d['summary'], author_str])

    print("training tfidf vectors...")
    v.fit(make_corpus())

    print("running inference...")
    x = v.transform(make_corpus()).astype(np.float32)
    print(x.shape)

    print("saving to features to disk...")
    features = {
        'pids': list(pdb.keys()),
        'x': x,
        'vocab': v.vocabulary_,
        'idf': v._tfidf.idf_,
    }
    save_features(features)
