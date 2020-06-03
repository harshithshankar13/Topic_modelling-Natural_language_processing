# -*- coding: utf-8 -*-

# install package
pip install GetOldTweets3

# install package
pip install pyLDAvis

# import 
import GetOldTweets3 as getOTw3
import pandas as pd

# import tweets from the galwaybayfmnews using GetOldTweets3
galwaybayfmTweetObj = getOTw3.manager.TweetCriteria().setQuerySearch('Galwaybayfmnews')\
                                           .setSince("2019-08-7")\
                                           .setUntil("2020-04-7")

# get tweets 
galwayBayFmTweets = getOTw3.manager.TweetManager.getTweets(galwaybayfmTweetObj)

# create dataframe from imported tweets
galwayBay_df = pd.DataFrame()

# store tweet into data frame
for tweet in galwayBayFmTweets:
   galwayBay_df = galwayBay_df.append(pd.Series(tweet.text), ignore_index=True)
   print(tweet.text)

import re
# remove all the punctuation in the tweets
galwayBay_df['process'] = galwayBay_df[0].map(lambda x: re.sub('[,\\.!?]', '', x))
# split string into array by space
galwayBay_df['process'] = galwayBay_df['process'].str.lower().str.split()

# build list of words from data frame
words = []
for word in galwayBay_df.loc[:,'process']:
  for sword in word:
    words.append(sword)

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

# define prefix words to remove from list of words
prefixes = ('http', 'https', 'news', 'galway','uhg','oughterard','new','says','it', 'galwayad','know','connemara', 'tuam' ,'we' ,'one', 'us','say','go','thanks','td','well','next','back','loughrea', 'see', 're', '&amp','may','ve','like','johnson', 'three' ,'two','near','that','na','ugh','galwayad','boris','galwaycoco','fyi','12','please','help')
# define suffix words to remove from list of words
suffix = ('galway', 'galwaybayfmnews', 'news',  'galwayad', '&amp')
# remove prefix and suffix
words = [x for x in words if not x.startswith(prefixes)]
words = [x for x in words if not x.endswith(suffix)]

# remove stopwords
words = [item for item in words if item not in ENGLISH_STOP_WORDS]

# to get matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer

# create tokenizer
countVectObj = CountVectorizer()

# get matrix of token counts of imported tweets
tokenData = countVectObj.fit_transform(words)

# Import LDA model from scikit learn 
from sklearn.decomposition import LatentDirichletAllocation

# total number of topics
num_of_topic = 20

# create lda model with 50 iterations and learning method is online
lda_model = LatentDirichletAllocation(n_components=num_of_topic, max_iter=50, learning_method='online')
lda_topics = lda_model.fit_transform(tokenData)

# visualise the topics in the cluster
def Visulise_topics(lda, features, numOfword):
    for top_idx, topic in enumerate(lda.components_):
        print("Topic %d:" % (top_idx))
        print(" ".join([features[i] for i in topic.argsort()[:-numOfword - 1:-1]]))

# get feature names from tokenizer
features = countVectObj.get_feature_names()
# visulise the topics
Visulise_topics(lda_model, features, 5)

# remove the irrelevant posts from the model's output
topicsPerDocu = pd.DataFrame(lda_topics, columns=["Topic"+str(i+1) for i in range(num_of_topic)])

# to compare all the topic probablity is equal in a document
val = 1/num_of_topic
# remove rows having all the topic probablities are equal
topicsPerDocu = topicsPerDocu.loc[(topicsPerDocu != val).all(axis=1),]

# select a topic with high probability as a topic of the perticular document
highProbTopic = topicsPerDocu.idxmax(axis=1)
# count number of documents per topic
docCountPerTopic = highProbTopic.groupby(highProbTopic).count()

# convert series into dataframe
docCountPerTopic_df = pd.DataFrame(docCountPerTopic)
docCountPerTopic_df.reset_index( inplace=True)
docCountPerTopic_df.columns = ["Topic Name", 'Number of Tweets']

# sort the dataframe in ascending order of the number of tweets
docCountPerTopic_df.sort_values(by=['Number of Tweets'], inplace=True)

# By manual selecting topics topics 1,3,13,15,17,8
docCountPerTopic_df = docCountPerTopic_df.loc[docCountPerTopic_df['Topic Name'].isin(['Topic1','Topic3','Topic8','Topic13','Topic15','Topic17'])]

docCountPerTopic_df

# save table to csv
docCountPerTopic_df.to_csv("tweetsPerTopic.csv", index=False)

# visualise document per topics
# bar plot
import matplotlib.pyplot as plot
plot.style.use('ggplot')

# define figure object
figure = plot.figure()
# define axes object
axes = figure.add_axes([0,0,1,1])

# visulise bar chart
plot.bar(docCountPerTopic_df['Topic Name'], docCountPerTopic_df['Number of Tweets'], width=0.4, color=['#009E73', "#CC79A7", "#E69F00", "#000000"])

# set title and ylabel
plot.ylabel('Number of tweets per topic')

# to visualise the LDA topic model using pyLDAvis 
import pyLDAvis.sklearn

pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda_model, tokenData, countVectObj, mds='tsne', R=5)
panel