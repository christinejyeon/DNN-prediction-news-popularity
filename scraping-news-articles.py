from bs4 import BeautifulSoup
import pandas
from urllib.request import urlopen

og = pandas.read_csv("/Users/Christine/Documents/GLIS 689/OnlineNewsPopularity_OnlineNewsPopularity.csv")
og_url = og.iloc[:,0]

article = ''
articles = pandas.DataFrame()
sigh = []
for i in range(len(og_url)):
    article = ''
    page = urlopen(og_url[i])
    soup = BeautifulSoup(page, 'html.parser')
    if soup.find('article', {"class": "full post story"}) is None:
        if soup.find('section', {"class": "article-content fullwidth"}) is None:
            content = soup.find('section', {"class": "article-content blueprint"})
        else:
            content = soup.find('section', {"class": "article-content fullwidth"})
    else:
        content = soup.find('article', {"class": "full post story"})
    try:
        for j in content.findAll('p'):
            article = article + ' ' + j.text
        article = [article]
        articles = articles.append(pandas.DataFrame(data=article.copy()))
    except:
        sigh.append(i)
