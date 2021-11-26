
# arxiv-sanity-lite


**(WIP)**

A much lighter-weight arxiv-sanity re-write. Currently runs only locally and doesn't exist as a website on the internet. However, the code is in a semi "feature-complete" state in the sense that you can look through arxiv papers, tag any of them arbitrarily, and then arxiv-sanity-lite recommends similar papers for each tag based on SVM on tfidf vectors constructed from the paper abstracts. So that's pretty cool, I find this personally plenty useful already, and it may be useful to you as well!

I hope to make this good over time and once it's ready to also host it publicly, deprecating the current bloated arxiv-sanity in favor of this new format. The biggest remaining todo's are adding user accounts and making everything nicer, faster, and more scalable as the number of papers in the database grows.

![Screenshot](screenshot.jpg)

#### To run

- (Periodically) run arxiv_daemon.py to add recent papers from arxiv to the database.
- Then run compute.py to re-calculate tfidf features on the paper abstracts and save those to database.
- Finally run serve.py to start the server and access the frontend layer over the data, e.g.: `export FLASK_APP=serve.py; flask run`.

#### todos

- add user accounts so we can shipit
- the metas table should not be a sqlitedict but a proper sqlite table, for efficiency
- build a reverse index to support faster search, right now we iterate through the entire database

#### License

MIT
