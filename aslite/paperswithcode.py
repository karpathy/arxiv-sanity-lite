import requests

def getGithubLink(arxiv_id):
    urls = []
    base_url = "https://paperswithcode.com/api/v1/"
    add_paper_id = "papers?arxiv_id=%s" %(arxiv_id)
    search_query = base_url + add_paper_id
    request = requests.get(search_query)
    if request:
        for _data in request.json()["results"]:
            _id = _data['id']
            add_id = "papers/%s/repositories/" %(_id)
            search_url = base_url + add_id
            request = requests.get(search_url)
            if request:
                for _urlInfo in request.json()["results"]:
                    if _urlInfo['is_official']:
                        urls.append(_urlInfo['url'])
            else:
                return None
    else:
        return None
    return urls
