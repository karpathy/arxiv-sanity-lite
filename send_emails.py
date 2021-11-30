"""
Compose and send recommendation emails to arxiv-sanity-lite users!

I run this script in a cron job to send out emails to the users with their
recommendations. There's a bit of copy paste code here but I expect that
the recommendations may become more complex in the future, so this is ok for now.

You'll notice that the file sendgrid_api_key.txt is not in the repo, you'd have
to manually register with sendgrid yourself, get an API key and put it in the file.
"""

import os
import time
import argparse

import numpy as np
from sklearn import svm

import sendgrid
from sendgrid.helpers.mail import Email, To, Content, Mail

from aslite.db import load_features
from aslite.db import get_tags_db
from aslite.db import get_metas_db
from aslite.db import get_papers_db
from aslite.db import get_email_db

# -----------------------------------------------------------------------------
# the html template for the email

template = """
<!DOCTYPE HTML>
<html>

<head>
<style>
body {
    font-family: Arial, sans-serif;
}
.s {
    font-weight: bold;
    margin-right: 10px;
}
.a {
    color: #333;
}
.u {
    font-size: 12px;
    color: #333;
    margin-bottom: 10px;
}
</style>
</head>

<body>

<br><br>
<div>Hi! Here are your <a href="https://arxiv-sanity-lite.com">arxiv-sanity-lite</a> recommendations. __STATS__</div>
<br><br>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To stop these emails remove your email in your <a href="https://arxiv-sanity-lite.com/profile">account</a> settings.
</div>
<div> <3, arxiv-sanity-lite. </div>

</body>
</html>
"""

# -----------------------------------------------------------------------------

def calculate_recommendation(
    tags,
    time_delta = 3, # how recent papers are we recommending? in days
    ):

    # a bit of preprocessing
    x, pids = features['x'], features['pids']
    n, d = x.shape
    ptoi, itop = {}, {}
    for i, p in enumerate(pids):
        ptoi[p] = i
        itop[i] = p

    # construct the positive set via simple union of all tags
    y = np.zeros(n, dtype=np.float32)
    for tag, pids in tags.items():
        for pid in pids:
            y[ptoi[pid]] = 1.0

    # classify
    clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
    clf.fit(x, y)
    s = clf.decision_function(x)
    sortix = np.argsort(-s)
    pids = [itop[ix] for ix in sortix]
    scores = [100*float(s[ix]) for ix in sortix]

    # filter by time to only recent papers
    deltat = time_delta*60*60*24 # allowed time delta in seconds
    keep = [i for i,pid in enumerate(pids) if (tnow - metas[pid]['_time']) < deltat]
    pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

    # finally exclude the papers we already have tagged
    have = set().union(*tags.values())
    keep = [i for i,pid in enumerate(pids) if pid not in have]
    pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

    return pids, scores

# -----------------------------------------------------------------------------

def render_recommendations(tags, pids, scores):
    # render the paper recommendations into the html template

    parts = []
    n = min(len(scores), args.num_recommendations)
    for score, pid in zip(scores[:n], pids[:n]):
        p = pdb[pid]
        authors = ', '.join(a['name'] for a in p['authors'])
        # crop the abstract
        summary = p['summary']
        summary = summary[:min(500, len(summary))]
        if len(summary) == 500:
            summary += '...'
        parts.append(
"""
<tr>
<td valign="top"><div class="s">%.2f</div></td>
<td>
<a href="%s">%s</a>
<div class="a">%s</div>
<div class="u">%s</div>
</td>
</tr>
""" % (score, p['link'], p['title'], authors, summary)
        )

    # render the final html
    out = template

    # render the recommendations
    final = '<table>' + ''.join(parts) + '</table>'
    out = out.replace('__CONTENT__', final)

    # render the stats
    num_papers_tagged = len(set().union(*tags.values()))
    stats = f"We took the {num_papers_tagged} papers across your {len(tags)} tags and \
              ranked {len(pids)} papers that showed up on arxiv over the last \
              {args.time_delta} days using tfidf SVMs over paper abstracts. Below are the \
              top {args.num_recommendations} papers. Remember that the more you tag, \
              the better this gets:"
    out = out.replace('__STATS__', stats)

    return out

# -----------------------------------------------------------------------------
# send the actual html via sendgrid

def send_email(to, html):

    # init the api
    assert os.path.isfile('sendgrid_api_key.txt')
    api_key = open('sendgrid_api_key.txt', 'r').read().strip()
    sg = sendgrid.SendGridAPIClient(api_key=api_key)

    # construct the email
    from_email = Email("admin@arxiv-sanity-lite.com")
    to_email = To(to)
    subject = tnow_str + " Arxiv Sanity Lite recommendations"
    content = Content("text/html", html)
    mail = Mail(from_email, to_email, subject, content)

    # hope for the best :)
    if not args.dry_run:
        response = sg.client.mail.send.post(request_body=mail.get())
        print(response.status_code)
        pass

# -----------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sends emails with recommendations')
    parser.add_argument('-n', '--num-recommendations', type=int, default=20, help='number of recommendations to send per person')
    parser.add_argument('-t', '--time-delta', type=int, default=3, help='how recent papers to recommended, in days')
    parser.add_argument('-d', '--dry-run', type=int, default=0, help='if set to 1 do not actually send the emails')
    parser.add_argument('-u', '--user', type=str, default='', help='restrict recommendations only to a single given user (used for debugging)')
    args = parser.parse_args()
    print(args)

    tnow = time.time()
    tnow_str = time.strftime('%b %d', time.localtime(tnow)) # e.g. "Nov 27"

    # read entire db simply into RAM
    with get_tags_db() as tags_db:
        tags = {k:v for k,v in tags_db.items()}

    # read entire db simply into RAM
    with get_metas_db() as mdb:
        metas = {k:v for k,v in mdb.items()}

    # read entire db simply into RAM
    with get_email_db() as edb:
        emails = {k:v for k,v in edb.items()}

    # read tfidf features into RAM
    features = load_features()

    # keep the papers as only a handle, since this can be larger
    pdb = get_papers_db()

    # iterate all users, create recommendations, send emails
    for user, tags in tags.items():

        # verify that we have an email for this user
        email = emails.get(user, None)
        if not email:
            print("skipping user %s, no email" % (user, ))
            continue
        if args.user and user != args.user:
            print("skipping user %s, not %s" % (user, args.user))
            continue

        # calculate the recommendations
        pids, scores = calculate_recommendation(tags, time_delta=args.time_delta)
        print("user %s has %d recommendations over last %d days" % (user, len(pids), args.time_delta))
        if len(pids) == 0:
            print("skipping the rest, no recommendations were produced")
            continue

        # render the html
        print("rendering top %d recommendations into a report..." % (args.num_recommendations, ))
        html = render_recommendations(tags, pids, scores)
        # temporarily for debugging write recommendations to disk for manual inspection
        if os.path.isdir('recco'):
            with open('recco/%s.html' % (user, ), 'w') as f:
                f.write(html)

        # actually send the email
        print("sending email...")
        send_email(email, html)


    print("done.")
