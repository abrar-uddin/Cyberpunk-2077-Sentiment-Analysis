import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sentiment analysis libraries
from textblob import TextBlob
import flair

st.title('Cyberpunk 2077 Sentiment Analysis')

# Image under title
image = Image.open('../images/comments_wordcloud.png')
st.markdown('by **Abrar Uddin** *Aspiring Data Scientist*')
'''
[Project GitHub](https://github.com/abrar-uddin/Cyberpunk-2077-Sentiment-Analysis)
'''
st.image(image, caption='Word cloud made from youtube comments', use_column_width=True)

st.markdown('The [Raw Data](https://www.kaggle.com/andrewmvd/cyberpunk-2077/data?select=comments.csv) for this '
            'project was provided by username Larxel on kaggle.')

st.markdown('# Model Demo')
'''
For this project I have used two pre-built NLP models from TextBlob and Flair. You can try out the capabilities of the
two models below, simply type in to the text box!
'''

# TextBlob Demo
textblob_demo = st.text_input('TextBlob Sentiment Analysis Demo', 'Cyberpunk is the best game of the decade!')
textblob_result = TextBlob(textblob_demo).sentiment
st.write('The polarity score for the inputted text is', textblob_result[0], "and the subjectivity score is",
         textblob_result[1])

'''
The polarity score ranges within [-1.0, 1.0] and the subjectivity score ranges within [0.0, 1.0]. If the polarity is 
negative [TextBlob](https://textblob.readthedocs.io/en/dev/) has assigned it a negative sentiment while the inverse 
would be positive with 0.0 being neutral. The subjectivity score is exactly what it means, it measures how subjective
the text is. 
'''

# Flair Demo
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

flair_demo = st.text_input('Flair Sentiment Analysis Demo', 'Cyberpunk is the best game of the decade!')
flair_input = flair.data.Sentence(flair_demo)
flair_sentiment.predict(flair_input)
flair_result = str(flair_input.labels[0]).replace('(', '').replace(')', '').split(' ')
st.write('The sentiment for the inputted text is', flair_result[0], 'with a score of', flair_result[1])

'''
With [Flair](https://github.com/flairNLP/flair) given the inputted text it spits back a binary label of POSITIVE or
NEGATIVE with a confidence score in the range of [0.0, 1.0]. 
'''

'''
# Dataset
'''
'''You can find the [cleaned dataset](https://github.com/abrar-uddin/Cyberpunk-2077-Sentiment-Analysis/blob/master/data/sentiment_analysis.csv) 
with the NLP classification already completed with the link. The dataset contains 171781 rows of comments scraped from
Youtube comments ranging from 2020-2012. For the pre-processing of the dataset I created a custom 
[text cleaning](https://github.com/abrar-uddin/Cyberpunk-2077-Sentiment-Analysis/blob/master/code/text_cleaner.py) tool. 
For the cleaning of the comments the text first had to be lower cased, then any URLs, Hashtags, Mentions, Reserved words 
(e.g. RT, FAV), Emojis, Smileys had to be removed. After which any digits in the text as well as punctuation were removed. 
Finally the text would be lemmatized and tokenized ready to be processed by the models. '''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
table = pd.read_sql_query("SELECT * FROM sentiment_analysis LIMIT 5", con)
con.close()
st.write(table,
         'Here is a quick look at the dataset. One obvious flaw of the pre-built models I would like to point is '
         'that it is unable to take in to consideration the context of the comments as such the accuracy of the'
         ' classifications are being impacted. This could be improved upon by training with labeled data however'
         ' such a task of labeling a large dataset such as this would be monumental for any single person.')

'''
# Visualization of findings
'''

# Pie Chart
model_results = pd.read_csv('../data/model_results.csv')
fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
fig.add_trace(go.Pie(labels=model_results['Sentiment'], values=model_results['textblob'], name="TextBlob"),
              1, 1)
fig.add_trace(go.Pie(labels=model_results['Sentiment'], values=model_results['flair'], name="Flair"),
              1, 2)
'### TextBlob vs Flair Classifications'
st.plotly_chart(fig)

# Bar Chart
fig = go.Figure(data=[
    go.Bar(name='TextBlob', x=model_results['Sentiment'], y=model_results['textblob']),
    go.Bar(name='Flair', x=model_results['Sentiment'], y=model_results['flair'])
])
fig.update_layout(barmode='group')
st.plotly_chart(fig)

# Texts with most likes
'''
### Top 3 Most Liked Comments
'''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
table = pd.read_sql_query("SELECT text,publishedAt,likeCount, textblob_polarity, flair_sentiment "
                          "FROM sentiment_analysis ORDER BY likecount DESC LIMIT 3", con)
con.close()
st.table(table)

# Texts with positive TextBlob polarity score
'''
### Top 3 Positive TextBlob Polarity Score
'''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
table = pd.read_sql_query(
    "SELECT text, likeCount, textblob_polarity, textblob_subjectivity, flair_sentiment, flair_score "
    "FROM sentiment_analysis ORDER BY textblob_polarity DESC LIMIT 3", con)
con.close()
st.table(table)

# Texts with positive Flair score
'''
### Top 3 Positive Flair Score
'''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
table = pd.read_sql_query(
    "SELECT text, likeCount, textblob_polarity, textblob_subjectivity, flair_sentiment, flair_score "
    "FROM sentiment_analysis ORDER BY flair_sentiment DESC, flair_score DESC LIMIT 3", con)
con.close()
st.table(table)

# Texts with negative TextBlob polarity score
'''
### Top 3 Negative TextBlob Polarity Score
'''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
table = pd.read_sql_query(
    "SELECT text, likeCount, textblob_polarity, textblob_subjectivity, flair_sentiment, flair_score "
    "FROM sentiment_analysis ORDER BY textblob_polarity ASC LIMIT 3", con)
con.close()
st.table(table)

# Texts with negative Flair score
'''
### Top 3 Negative Flair Score
'''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
table = pd.read_sql_query(
    "SELECT text, likeCount, textblob_polarity, textblob_subjectivity, flair_sentiment, flair_score "
    "FROM sentiment_analysis ORDER BY flair_sentiment ASC, flair_score ASC LIMIT 3", con)
con.close()
st.table(table)

# Texts with longest comment length
'''
### Top 2 Longest comments
'''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
table = pd.read_sql_query("SELECT text, comment_length "
                          "FROM sentiment_analysis ORDER BY comment_length DESC LIMIT 2", con)
con.close()
st.write(table)

# Sentiment Analysis From 2012-2020
'''
### Sentiment Analysis From 2012-2020
'''
def query_db(num1, num2=None, flip=False):
    if num2 is None:
        text = "SELECT textblob_polarity, flair_sentiment FROM sentiment_analysis \
                WHERE publishedAt < {}".format(num1)
    elif flip is True:
        text = "SELECT textblob_polarity, flair_sentiment FROM sentiment_analysis \
                WHERE publishedAt > {}".format(num1)
    else:
        num2 = int(num2)
        text = "SELECT textblob_polarity, flair_sentiment FROM sentiment_analysis \
                WHERE publishedAt > {} AND publishedAt < {}".format(num1, num2)

    return text


labels = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
textblob_positives = []
flair_positives = []

textblob_neutral = []

textblob_negatives = []
flair_negatives = []

con = sqlite3.connect("sentiment_analysis_db.sqlite")
for x in labels:
    textblob_results = {"positive": 0, "neutral": 0, "negative": 0}
    if labels.index(x) == 2012:
        table = pd.read_sql_query(query_db(x), con)
    elif labels.index(x) == 2020:
        table = pd.read_sql_query(query_db(x, flip=True), con)
    else:
        table = pd.read_sql_query(query_db(x, int(x) + 1), con)

    for x in table['textblob_polarity']:
        if x == 0.0:
            textblob_results["neutral"] += 1
        elif x > 0.0:
            textblob_results["positive"] += 1
        else:
            textblob_results["negative"] += 1
    textblob_positives.append(textblob_results.get('positive'))
    flair_positives.append(table['flair_sentiment'].value_counts()['POSITIVE'])

    textblob_neutral.append(textblob_results.get('neutral'))

    textblob_negatives.append(textblob_results.get('negative'))
    flair_negatives.append(table['flair_sentiment'].value_counts()['NEGATIVE'])
con.close()

'#### TextBlob'
fig = go.Figure(data=[
    go.Bar(name='Positive', x=labels, y=textblob_positives),
    go.Bar(name='Neutral', x=labels, y=textblob_neutral),
    go.Bar(name='Negative', x=labels, y=textblob_negatives)
])
# Change the bar mode
fig.update_layout(barmode='group')
st.plotly_chart(fig)

'#### Flair'
fig = go.Figure(data=[
    go.Bar(name='Positive', x=labels, y=flair_positives),
    go.Bar(name='Neutral', x=labels, y=[]),
    go.Bar(name='Negative', x=labels, y=flair_negatives)
])
# Change the bar mode
fig.update_layout(barmode='group')
st.plotly_chart(fig)