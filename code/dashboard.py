import streamlit as st
import pandas as pd
import sqlite3
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Sentiment analysis libraries
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence

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
@st.cache
def flair_classifier():
    return TextClassifier.load('en-sentiment')


flair_sentiment = flair_classifier()

flair_demo = st.text_input('Flair Sentiment Analysis Demo', 'Cyberpunk is the best game of the decade!')
flair_input = Sentence(flair_demo)
flair_sentiment.predict(flair_input)
flair_result = str(flair_input.labels[0]).replace('(', '').replace(')', '').split(' ')
st.write('The sentiment for the inputted text is', '**'+flair_result[0]+'**', 'with a score of', float(flair_result[1]))

'''
With [Flair](https://github.com/flairNLP/flair) given the inputted text it spits back a binary label of POSITIVE or
NEGATIVE with a confidence score in the range of [0.0, 1.0]. 
'''


@st.cache
# Function to read CSVs
def read_csv(path):
    return pd.read_csv(path)


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
model_results = read_csv('../data/model_results.csv')
fig = make_subplots(rows=2, cols=2,
                    specs=[[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'xy', "colspan": 2}, None]],
                    subplot_titles=("TextBlob", "Flair"))
fig.add_trace(
    go.Pie(labels=model_results['Sentiment'], values=model_results['textblob'], name="TextBlob", showlegend=False),
    1, 1)
fig.add_trace(go.Pie(labels=model_results['Sentiment'], values=model_results['flair'], name="Flair", showlegend=False),
              1, 2)
fig.add_trace(go.Bar(name='TextBlob', x=model_results['Sentiment'], y=model_results['textblob']), 2, 1)
fig.add_trace(go.Bar(name='Flair', x=model_results['Sentiment'], y=model_results['flair']), 2, 1)

fig.update_layout(height=700, legend=dict(
    yanchor="top",
    y=.37,
    xanchor="left",
    x=1
), title_text='TextBlob vs Flair Sentiment Classifications')
st.plotly_chart(fig)


# Sentiment Analysis From 2012-2020
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

    positive = table['textblob_polarity'] > 0
    neutral = table['textblob_polarity'] == 0
    negative = table['textblob_polarity'] < 0

    textblob_positives.append(positive.value_counts()[1])
    flair_positives.append(table['flair_sentiment'].value_counts()['POSITIVE'])

    textblob_neutral.append(neutral.value_counts()[1])

    textblob_negatives.append(negative.value_counts()[1])
    flair_negatives.append(table['flair_sentiment'].value_counts()['NEGATIVE'])
con.close()

fig = make_subplots(rows=1, cols=1,
                    specs=[[{'type': 'xy'}]])
fig.add_trace(go.Bar(name='TextBlob: Positive', x=labels, y=textblob_positives), 1, 1)
fig.add_trace(go.Bar(name='TextBlob: Neutral', x=labels, y=textblob_neutral), 1, 1)
fig.add_trace(go.Bar(name='TextBlob: Negative', x=labels, y=textblob_negatives), 1, 1)

fig.add_trace(go.Bar(name='Flair: Positive', x=labels, y=flair_positives), 1, 1)
fig.add_trace(go.Bar(name='Flair: Negative', x=labels, y=flair_negatives), 1, 1)
fig.update_layout(height=500, width=750, title_text='Sentiment Analysis From 2012-2020')
st.plotly_chart(fig)

# Google Trends
google_trends = read_csv('../data/raw_data/google_trends_cyberpunk.csv')
fig = go.Figure(data=go.Scatter(name='Google Trends', x=google_trends['year-month'], y=google_trends['cyberpunk 2077']))
fig.update_layout(title_text='Google Trends')
st.plotly_chart(fig)

# Video Statistics
'''
### Statistics by Video
'''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
index = pd.read_sql_query("SELECT videoId "
                          "FROM sentiment_analysis GROUP BY videoId", con)
table1 = pd.read_sql_query("SELECT COUNT(textblob_polarity) as TextBlob_Positive "
                           "FROM sentiment_analysis WHERE textblob_polarity > 0 GROUP BY videoId", con)
table2 = pd.read_sql_query("SELECT COUNT(textblob_subjectivity) as TextBlob_Neutral "
                           "FROM sentiment_analysis WHERE textblob_subjectivity == 0 GROUP BY videoId", con)
table3 = pd.read_sql_query("SELECT COUNT(textblob_polarity) as TextBlob_Negative "
                           "FROM sentiment_analysis WHERE textblob_polarity < 0 GROUP BY videoId", con)

table4 = pd.read_sql_query("SELECT COUNT(flair_score) as Flair_Positive "
                           "FROM sentiment_analysis WHERE flair_sentiment == \"POSITIVE\" GROUP BY videoId", con)
table5 = pd.read_sql_query("SELECT COUNT(flair_score) as Flair_Negative "
                           "FROM sentiment_analysis WHERE flair_sentiment == \"NEGATIVE\" GROUP BY videoId", con)
con.close()
frames = [index, table1, table2, table3, table4, table5]
frames = pd.concat(frames, axis=1)
vid_table = read_csv('../data/raw_data/videos.csv')
table = frames.merge(vid_table, left_on='videoId', right_on='videoId').drop(['categoryId', 'channelId', 'description'],
                                                                            axis=1).sort_values(
    ['publishedAt']).set_index('videoId')
# Video Stats
fig = make_subplots(rows=2, cols=2,
                    specs=[[{'type': 'xy'}, {'type': 'xy'}], [{'type': 'xy', "colspan": 2}, None]],
                    subplot_titles=("View Count vs Comment Count", "Like vs Dislike", "Percent of Viewers Commented"))
fig.add_trace(go.Bar(name='View Count', x=table['title'], y=table['viewCount']), 1, 1)
fig.add_trace(go.Bar(name='Comment Count', x=table['title'], y=table['commentCount']), 1, 1)
fig.add_trace(go.Bar(name='Like Count', x=table['title'], y=table['likeCount']), 1, 2)
fig.add_trace(go.Bar(name='Dislike Count', x=table['title'], y=table['dislikeCount']), 1, 2)
fig.add_trace(go.Bar(name='Percent Engagement', x=table['title'],
                     y=[100 * (x / y) for x, y in zip(table['commentCount'], table['viewCount'])], showlegend=False), 2,
              1)

fig.update_layout(title_text='CyberPunk Video Performance', barmode='stack', height=600, width=800)
fig.update_xaxes(showticklabels=False)
st.plotly_chart(fig)

# Sentiment by Video
fig = make_subplots(rows=1, cols=1,
                    specs=[[{'type': 'xy'}]])
# TextBlob
fig.add_trace(go.Bar(name='TextBlob: Positive', x=table['title'], y=table['TextBlob_Positive']), 1, 1)
fig.add_trace(go.Bar(name='TextBlob: Neutral', x=table['title'], y=table['TextBlob_Neutral']), 1, 1)
fig.add_trace(go.Bar(name='TextBlob: Negative', x=table['title'], y=table['TextBlob_Negative']), 1, 1)
# Flair
fig.add_trace(go.Bar(name='Flair: Positive', x=table['title'], y=table['Flair_Positive']), 1, 1)
fig.add_trace(go.Bar(name='Flair: Negative', x=table['title'], y=table['Flair_Negative']), 1, 1)
fig.update_layout(title_text='CyberPunk Video Sentiments', height=700, width=1000)
fig.update_xaxes(showticklabels=False)
st.plotly_chart(fig)

for i in range(19):
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                        subplot_titles=("TextBlob", "Flair"))  # subplot_titles=(table['title'])
    fig.add_trace(
        go.Pie(labels=model_results['Sentiment'],
               values=table[['TextBlob_Positive', 'TextBlob_Neutral', 'TextBlob_Negative']].iloc[i], name="TextBlob",
               showlegend=False),
        1, 1)
    fig.add_trace(
        go.Pie(labels=model_results['Sentiment'].drop(1), values=table[['Flair_Positive', 'Flair_Negative']].iloc[i],
               name="Flair", showlegend=False),
        1, 2)
    fig.update_layout(title_text=table['title'][i])
    st.plotly_chart(fig)

st.write(table)

# Texts with most likes
'''
### Top 5 Most Liked Comments
'''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
table = pd.read_sql_query("SELECT text,likeCount "
                          "FROM sentiment_analysis ORDER BY likecount DESC LIMIT 5", con)
con.close()
st.table(table)

# Texts with positive TextBlob polarity score
'''
### Top 5 Positive TextBlob Polarity Score
'''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
table = pd.read_sql_query(
    "SELECT text, textblob_polarity, textblob_subjectivity "
    "FROM sentiment_analysis ORDER BY textblob_polarity DESC LIMIT 5", con)
con.close()
st.table(table)

# Texts with positive Flair score
'''
### Top 5 Positive Flair Score
'''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
table = pd.read_sql_query(
    "SELECT text, flair_sentiment, flair_score "
    "FROM sentiment_analysis ORDER BY flair_sentiment DESC, flair_score DESC LIMIT 5", con)
con.close()
st.table(table)

# Texts with negative TextBlob polarity score
'''
### Top 5 Negative TextBlob Polarity Score
'''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
table = pd.read_sql_query(
    "SELECT text, textblob_polarity, textblob_subjectivity "
    "FROM sentiment_analysis ORDER BY textblob_polarity ASC LIMIT 5", con)
con.close()
st.table(table)

# Texts with negative Flair score
'''
### Top 5 Negative Flair Score
'''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
table = pd.read_sql_query(
    "SELECT text, flair_sentiment, flair_score "
    "FROM sentiment_analysis ORDER BY flair_sentiment ASC, flair_score DESC LIMIT 5", con)
con.close()
st.table(table)

# Texts with longest comment length
'''
### Top 5 Longest comments
'''
con = sqlite3.connect("sentiment_analysis_db.sqlite")
table = pd.read_sql_query("SELECT text, comment_length "
                          "FROM sentiment_analysis ORDER BY comment_length DESC LIMIT 5", con)
con.close()
st.table(table.style.format({"text": lambda z: z[0:40] + '...'}))

'''
# Conclusion 
From analyzing the charts we can see that less than 1% of the viewership engages with the video. From 
those that do decide to comment we can infer that they tend to feel strongly about the game as commenters are in the 
minority. The classified sentiments from Flair tends to be much more accurate than TextBlob as we can see from the 
comparisons above; the Top 5 Negative TextBlob sentiments are 4/5 false positive. As such we will only consider the 
Flair sentiment in our conclusions. Flair tends to be more negative than positive, however we need to look at this 
with a bit of skepticism as we saw from the Top 5 negative comments there can be false negatives. Also taking into 
consideration the fact that the videos tend to have very low dislike values we can take that as an indicator that 
viewers who felt strongly negative after watching the video chose to comment versus those who enjoyed the video but 
only chose to hit the like button. Tasking a look at the most positive comments they tend to be high quality genuine 
admiration for the game. Although there are a minority who dislike the video the negative comments tend to make up 
most of the comment section or so **I thought**. Taking a closer look at the comments that were classified negative 
it turns out there may be more **objectively** negative comment in other words false positives. Comments with 
negative words but are intended in a positive sentiment. Taking a closer look at the TextBlob classifications as well 
we see a similar situation. This is an inherent flaw in using a pre-trained model as it lacks the contextual learning 
from supervised methods that can prevent this kinds of misclassifications. For example a comment that simply states 
"KEANUUUUUU" is classified as negative with a score of 0.5006 under Flair. From the score we can see that the model 
is unsure of this classification and is guessing it is negative. The lesson to gleam from this is that gamers seem to 
make positive comments with negative sentiments. '''
