import pandas
import numpy

# Universal sentence encoder
import tensorflow as tf
import tensorflow_hub as hub
def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})

embed_fn = embed_useT("https://tfhub.dev/google/universal-sentence-encoder/2")

# embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

og_dataset = pandas.read_csv("/home/christinejyeon/newspop/og_dataset.csv")
# og_dataset = pandas.read_csv("/Users/Christine/Documents/GLIS 689/og_dataset.csv")
og_dataset = og_dataset.drop(columns=["Unnamed: 0"],axis=1)

og_dataset["vec"] = numpy.nan
og_dataset["vec"] = og_dataset["vec"].astype(object)
for i in range(len(og_dataset)):
    try:
        og_dataset.at[i,"vec"] = embed_fn([og_dataset.iloc[i,0]])
    except:
        og_dataset.at[i,"vec"] = numpy.nan

# embed_fn([og_dataset.iloc[0,0]])


# og_dataset.to_csv("/home/christinejyeon/newspop/og_dataset_vec.csv",sep='\t', encoding='utf-8')
# og_dataset.to_pickle("/home/christinejyeon/newspop/og_dataset_vec.pkl")
# og_dataset.to_excel("/home/christinejyeon/newspop/og_dataset_vec.xlsx")

