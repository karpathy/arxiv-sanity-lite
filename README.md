
# arxiv-sanity-lite


**(WIP)**

A much lighter-weight arxiv-sanity re-write. Currently only runs locally on a single machine and doesn't actually exist as a website on the internet. However, the code is currently in a semi "feature-complete" state in the sense that I can personally run it locally on my computer and find it helpful to me. Basically I find the papers that look good and use the UI to tag them under any category of interest. Then the code recommends other similar papers for each tag based on SVM on tfidf vectors constructed from abstracts. So that's pretty cool, and may be useful to you as well!

That said, the code was written quick & dirty style, so one currently has to read it and you're on your own wrt any support. But I hope to make it good and host it publicly in the future, deprecating the current bloated arxiv-sanity in favor of this format.


#### To run

- Periodically run arxiv_daemon.py to add recent papers from arxiv to the database.
- Then run compute.py to calculate tfidf features on the paper abstracts and save those to database.
- Finally run serve.py to start the server and access the frontend layer over the data, e.g.: `export FLASK_APP=serve.py; flask run`.


#### License

MIT