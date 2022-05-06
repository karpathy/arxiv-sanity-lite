'use strict';

const Tweet = props => {
    const p = props.tweet;
    // tweet, score
    return (
        twttr.widgets.createTweet(
            String(p.id),
            document.getElementById('tweetwrap')
          )
    )
}

const TweetList = props => {
    const lst = props.tweets;
    const tweets_desc = props.tweets_desc;
    const tlst = lst.map((jtweet, ix) => <Tweet key={ix} tweet={jtweet} />);
    return (
        <div>
            <div>{tweets_desc}</div>
            <div id="tweetList" class="rel_tweets">
                {tlst}
            </div>
        </div>
    )
}

var elt = document.getElementById('tweetwrap');
if(elt) {
    ReactDOM.render(<TweetList tweets={tweets} tweets_desc={tweets_desc} />, elt);
}
