import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
import re
from nltk.corpus import stopwords
import string

# Load the data
data = pd.read_csv('filename.csv', encoding='latin1')
print(data.head())

# Check for missing values
data = data[["username", "tweet", "language"]]
print(data.isnull().sum())

#count the number of tweets in each language
print(data["language"].value_counts())

#common stop words include: a, the, and , or , of , on , this , we , were,...
nltk.download('stopwords')
#Snowball is a small string processing language for creating stemming algorithms for use in Information Retrieval,
# plus a collection of stemming algorithms implemented using it.
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))


def clean(text):
    text = str(text).lower()
    #replace the following with space
    # text = re.sub('\[.*?\]', '', text)
    # text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    # text = re.sub('\w*\d\w*', '', text)
    #remove stop words
    #remove stemming
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

data["tweet"] = data["tweet"].apply(clean)


text = " ".join(i for i in data.tweet)

stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

#vader is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media
nltk.download('vader_lexicon')
#SentimentIntensityAnalyzer is a class that is used to analyze the sentiment of text data.
sentiments = SentimentIntensityAnalyzer()
#polarity_scores is a method that is used to calculate the polarity of the text data.
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["tweet"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["tweet"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["tweet"]]
data = data[["tweet", "Positive", "Negative", "Neutral"]]
print(data.head())