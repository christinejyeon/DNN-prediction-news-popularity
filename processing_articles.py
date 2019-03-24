import pandas

og = pandas.read_csv("/Users/Christine/Documents/GLIS 689/OnlineNewsPopularity_OnlineNewsPopularity.csv")
og_articles = pandas.read_csv("/Users/Christine/Documents/GLIS 689/og_articles.csv")
omitted_articles = pandas.read_csv("/Users/Christine/Documents/GLIS 689/omitted_articles.csv")

omitted_articles = omitted_articles.drop("Unnamed: 0", axis=1)
omitted_articles.columns = ["indexno"]
omitted_articles["indexno"].tolist()

