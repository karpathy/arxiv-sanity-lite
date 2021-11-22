'use strict';

const PaperLite = props => {
    const p = props.paper;
    return (
    <div class='rel_paper'>
        <div class='rel_title'><a href={'http://arxiv.org/abs/' + p.id}>{p.title}</a></div>
        <div class='rel_authors'>{p.authors}</div>
        <div class="rel_time">{p.time}</div>
        <div class='rel_tags'>{p.tags}</div>
        <div class='rel_abs'>{p.summary}</div>
    </div>
    )
}


const Word = props => {
    const p = props.word;
    // word, weight, idf
    return (
    <div class='rel_word'>
        <div class='rel_word_weight'>{p.weight.toFixed(2)}</div>
        {/* <div class='rel_word_idf'>{p.idf.toFixed(2)}</div> */}
        <div class="rel_word_txt">{p.word}</div>
    </div>
    )
}

const WordList = props => {
    const lst = props.words;
    const wlst = lst.map((jword, ix) => <Word key={ix} word={jword} />);
    return (
        <div>
            <div>The following are the tokens and their (tfidf) weight in the paper vector. This is the actual summary that feeds into the SVM to power recommendations, so hopefully it is good and representative!</div>
            <div id="wordList" class="rel_words">
                {wlst}
            </div>
        </div>
    )
}

ReactDOM.render(<PaperLite paper={paper} />, document.getElementById('wrap'))
ReactDOM.render(<WordList words={words} />, document.getElementById('wordwrap'))
