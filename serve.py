"""
Flask server backend

ideas:
- allow delete of tags
- unify all different pages into single search filter sort interface
- special single-image search just for paper similarity
"""

import time
import pickle
from random import shuffle

import numpy as np
from sklearn import svm

from flask import Flask, request, redirect, url_for
from flask import render_template
from flask import g # global session-level object

from aslite.db import SqliteDict, CompressedSqliteDict

# -----------------------------------------------------------------------------
# TODO: user accounts / password login are necessary...

app = Flask(__name__)
RET_NUM = 100 # number of papers to return per page

def get_tags():
    if not hasattr(g, '_tags'):
        user = 'root' # root for now, the only default user
        print("reading tags for user %s" % (user, ))
        with CompressedSqliteDict('dict.db', tablename='tags', flag='r') as dict_db:
            tags_dict = dict_db[user] if user in dict_db else {}
        g._tags = tags_dict
    return g._tags

def get_papers():
    if not hasattr(g, '_pdb'):
        g._pdb = CompressedSqliteDict('papers.db', tablename='papers', flag='r')
    return g._pdb

def get_metas():
    if not hasattr(g, '_mdb'):
        g._mdb = SqliteDict('papers.db', tablename='metas', flag='r')
    return g._mdb

def render_pids(pids):

    pdb = get_papers()
    tags = get_tags()

    papers = []
    for pid in pids:
        d = pdb[pid]
        ptags = [t for t, pids in tags.items() if pid in pids]
        papers.append({
            'weight': 0.0,
            'id': d['_id'],
            'title': d['title'],
            'time': d['_time_str'],
            'authors': ', '.join(a['name'] for a in d['authors']),
            'tags': ', '.join(t['term'] for t in d['tags']),
            'utags': ptags,
            'summary': d['summary'],
        })

    return papers

def random_rank():
    pdb = get_papers()
    pids = list(pdb.keys())
    shuffle(pids)
    scores = [0 for _ in pids]
    return pids, scores

def time_rank():
    mdb = get_metas()
    ms = sorted(mdb.items(), key=lambda kv: kv[1]['_time'], reverse=True)
    tnow = time.time()
    pids = [k for k, v in ms]
    scores = [(tnow - v['_time'])/60/60/24 for k, v in ms] # time delta in days
    return pids, scores

def svm_rank(tags=None, pid=None):

    # tag can be one tag or a few comma-separated tags or 'all' for all tags we have in db
    # pid can be a specific paper id to set as positive for a kind of nearest neighbor search
    assert (tags is not None) or (pid is not None)

    # load all of the features
    features = pickle.load(open('features.p', 'rb'))
    x, pids = features['x'], features['pids']
    n, d = x.shape
    ptoi, itop = {}, {}
    for i, p in enumerate(pids):
        ptoi[p] = i
        itop[i] = p

    # construct the positive set
    y = np.zeros(n, dtype=np.float32)
    if pid is not None:
        y[ptoi[pid]] = 1.0
    elif tags is not None:
        tags_db = get_tags()
        tags_filter_to = tags_db.keys() if tags == 'all' else set(tags.split(','))
        for tag, pids in tags_db.items():
            if tag in tags_filter_to:
                for pid in pids:
                    y[ptoi[pid]] = 1.0

    if y.sum() == 0:
        return [], [] # there are no positives?

    # classify
    clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
    clf.fit(x, y)
    s = clf.decision_function(x)
    sortix = np.argsort(-s)
    pids = [itop[ix] for ix in sortix]
    scores = [100*float(s[ix]) for ix in sortix]

    return pids, scores

def default_context(papers, **kwargs):
    context = {}
    # insert the papers
    context['papers'] = papers
    # fetch and insert the available tags
    tags = get_tags()
    context['tags'] = [{'name':t, 'n':len(pids)} for t, pids in tags.items()] + [{'name': 'all'}]
    # various other globals
    gvars = {}
    gvars['search_query'] = ''
    gvars['time_filter'] = ''
    gvars['message'] = 'default_message'
    context['gvars'] = gvars
    return context

# -----------------------------------------------------------------------------

@app.teardown_request
def close_connection(error=None):
    # close any opened database connections
    if hasattr(g, '_pdb'):
        g._pdb.close()
    if hasattr(g, '_mdb'):
        g._mdb.close()

# -----------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def main():

    # GET options
    opt_rank = request.args.get('rank', 'time') # rank type. tags|pid|time|random
    opt_tags = request.args.get('tags', 'all')  # tags to rank by if opt_rank == 'tag'
    opt_pid = request.args.get('pid', None)  # pid to find nearest neighbors to
    opt_time_filter = request.args.get('time_filter', '') # number of days to filter by
    opt_skip_have = request.args.get('skip_have', 'no') # hide papers we already have?

    # rank papers: by tags, by time, by random
    if opt_rank in ['tags', 'pid']:
        pids, scores = svm_rank(tags=opt_tags, pid=opt_pid)
    elif opt_rank == 'time':
        pids, scores = time_rank()
    elif opt_rank == 'random':
        pids, scores = random_rank()
    else:
        raise ValueError("opt_rank %s is not a thing" % (opt_rank, ))

    # filter by time
    if opt_time_filter:
        pdb = get_papers()
        tnow = time.time()
        deltat = int(opt_time_filter)*60*60*24 # allowed time delta in seconds
        keep = [i for i,pid in enumerate(pids) if (tnow - pdb[pid]['_time']) < deltat]
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

    # optionally hide papers we already have
    if opt_skip_have == 'yes':
        tags_db = get_tags()
        have = set().union(*tags_db.values())
        keep = [i for i,pid in enumerate(pids) if pid not in have]
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

    # crop
    pids = pids[:min(len(pids), RET_NUM)]
    papers = render_pids(pids)
    for i, p  in enumerate(papers):
        p['weight'] = float(scores[i])

    context = default_context(papers)
    context['gvars']['rank'] = opt_rank
    context['gvars']['tags'] = opt_tags
    context['gvars']['time_filter'] = opt_time_filter
    return render_template('index.html', **context)


@app.route("/search", methods=['GET'])
def search():
    q = request.args.get('q', '') # get the search request
    if not q:
        return redirect(url_for('main')) # if someone just hits enter with empty field
    qs = q.lower().strip().split() # split by spaces

    match = lambda s: sum(s.lower().count(qp) for qp in qs)
    pairs = []
    pdb = get_papers()
    for pid, p in pdb.items():
        score = 0.0
        score += 5.0 * match(' '.join([a['name'] for a in p['authors']]))
        score += 10.0 * match(p['title'])
        score += 1.0 * match(p['summary'])
        if score > 0:
            pairs.append((score, pid))

    pairs.sort(reverse=True)
    pids = [p[1] for p in pairs]
    pids = pids[:min(RET_NUM, len(pids))] # crop if needed

    papers = render_pids(pids)
    for i, p in enumerate(papers):
        p['weight'] = pairs[i][0]

    context = default_context(papers)
    context['gvars']['search_query'] = q
    return render_template('index.html', **context)


@app.route('/add/<pid>/<tag>')
def add(pid=None, tag=None):
    user = 'root'
    with CompressedSqliteDict('dict.db', tablename='tags', flag='c') as dict_db:

        # create the user if we don't know about them yet with an empty library
        if not user in dict_db:
            dict_db[user] = {}

        # fetch the user library object
        d = dict_db[user]

        # add the paper to the tag
        if tag not in d:
            d[tag] = set()
        d[tag].add(pid)

        # write back to database
        dict_db[user] = d
        dict_db.commit()

    print("added paper %s to tag %s for user %s" % (pid, tag, user))
    return "ok: " + str(d) # return back the user library for debugging atm

@app.route('/del/<tag>')
def delete_tag(tag=None):
    user = 'root'
    with CompressedSqliteDict('dict.db', tablename='tags', flag='c') as dict_db:

        if user not in dict_db:
            return "user does not have a library"

        d = dict_db[user]

        if tag not in d:
            return "user does not have this tag"

        # delete the tag
        del d[tag]

        # write back to database
        dict_db[user] = d
        dict_db.commit()

    print("deleted tag %s for user %s" % (tag, user))
    return "ok: " + str(d) # return back the user library for debugging atm
