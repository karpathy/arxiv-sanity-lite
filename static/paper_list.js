'use strict';

const UTag = props => {
    const tag_name = props.tag;
    const turl = "/?rank=tags&tags=" + tag_name;
    return (
        <div class='rel_utag'>
            <a href={turl}>
                {tag_name}
            </a>
        </div>
    )
}

const Paper = props => {
    const p = props.paper;
    const adder = () => fetch("/add/" + p.id + "/" + prompt("tag name:"))
                        .then(response => console.log(response.text()));
    const utags = p.utags.map((utxt, ix) => <UTag key={ix} tag={utxt} />);
    const similar_url = "/?rank=tags&pid=" + p.id;
    return (
    <div class='rel_paper'>
        <div class="rel_add" onClick={adder}>+</div>
        <div class="rel_score">{p.weight.toFixed(2)}</div>
        <div class='rel_title'><a href={'http://arxiv.org/abs/' + p.id}>{p.title}</a></div>
        <div class='rel_authors'>{p.authors}</div>
        <div class="rel_time">{p.time}</div>
        <div class='rel_tags'>{p.tags}</div>
        <div class='rel_utags'>{utags}</div>
        <div class='rel_abs'>{p.summary}</div>
        <div class='rel_more'><a href={similar_url}>similar</a></div>
    </div>
    )
}

const PaperList = props => {
    const lst = props.papers;
    const plst = lst.map((jpaper, ix) => <Paper key={ix} paper={jpaper} />);
    return (
        <div>
            <div id="paperList" class="rel_papers">
                {plst}
            </div>
        </div>
    )
}

const Tag = props => {
    const t = props.tag;
    const turl = "/?rank=tags&tags=" + t.name;
    return (
        <div class='rel_utag'>
            <a href={turl}>
                {t.n} {t.name}
            </a>
        </div>
    )
}

const TagList = props => {
    const lst = props.tags;
    const tlst = lst.map((jtag, ix) => <Tag key={ix} tag={jtag} />);
    const deleter = () => fetch("/del/" + prompt("delete tag name:"))
                          .then(response => console.log(response.text()));
    return (
        <div>
            <div class="rel_del" onClick={deleter}>-</div>
            <div id="tagList" class="rel_utags">
                {tlst}
            </div>
        </div>
    )
}

const Opts = props => {
    const g = props.gvars;
    return (
        <div>
             time filter (days): <input type="text" value={g.time_filter} />
        </div>
    )
}

ReactDOM.render(<PaperList papers={papers} />, document.getElementById('wrap'))
ReactDOM.render(<TagList tags={tags} />, document.getElementById('tagwrap'))
//ReactDOM.render(<Opts gvars={gvars} />, document.getElementById('cbox'))