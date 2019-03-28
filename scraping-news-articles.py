from bs4 import BeautifulSoup
import pandas
from urllib.request import urlopen

og = pandas.read_csv("/Users/Christine/Documents/GLIS 689/OnlineNewsPopularity_OnlineNewsPopularity.csv")
og_url = og.iloc[:,0]

article = ''
articles = pandas.DataFrame()
articles_diff = []
sigh = []


for i in range(len(og_url)):
    article = ''
    try:
        page = urlopen(og_url[i])
        soup = BeautifulSoup(page, 'html.parser')
        if soup.find('article', {"class": "full post story"}) is None:
            if soup.find('section', {"class": "article-content fullwidth"}) is None:
                if soup.find('section', {"class": "article-content blueprint"}) is None:
                    content = soup.find('section', {"class": "article-content viral-video"})
                else:
                    content = soup.find('section', {"class": "article-content blueprint"})
            else:
                content = soup.find('section', {"class": "article-content fullwidth"})
        else:
            content = soup.find('article', {"class": "full post story"})
        for j in content.findAll('p'):
            article = article + ' ' + j.text
        article = [article]
        articles = articles.append(pandas.DataFrame(data=article.copy()))
    except:
        sigh.append(i)


og_articles = articles.copy()
og_articles.to_csv("og_articles.csv")

omitted_articles = pandas.DataFrame(sigh)
omitted_articles.to_csv("omitted_articles.csv")




## Building an ultimate dataset

og = pandas.read_csv("/Users/Christine/Documents/GLIS 689/OnlineNewsPopularity_OnlineNewsPopularity.csv")
og_articles = pandas.read_csv("/Users/Christine/Documents/GLIS 689/og_articles.csv")
omitted_articles = pandas.read_csv("/Users/Christine/Documents/GLIS 689/omitted_articles.csv")

omitted_articles = omitted_articles.drop("Unnamed: 0", axis=1)
omitted_articles.columns = ["indexno"]
omitted_articles = omitted_articles["indexno"].tolist()

og = og.drop(omitted_articles, axis=0)

og_articles = og_articles.drop("Unnamed: 0", axis=1)
og_dataset = pandas.concat([og_articles.reset_index(drop=True), og.drop("url", axis=1).reset_index(drop=True)], axis=1)

# Scraped articles + its popularity features
og_dataset.to_csv("og_dataset.csv")

