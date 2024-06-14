import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv('Twitter-Sentiment.csv')

# Print the column names to identify the correct text column
print("Column Names in the Dataset:")
print(df.columns)

# Replace 'actual_column_name' with the correct column name for the text data in your dataset
text_column = 'Borderlands'  # Replace 'Borderlands' with the correct column name

# Display the first few rows of the dataset
print("\nInitial Dataset:")
print(df.head())

# Preprocess the text data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stop words and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing
df['clean_text'] = df[text_column].apply(preprocess_text)

# Display the first few rows of the cleaned dataset
print("\nCleaned Dataset:")
print(df[[text_column, 'clean_text']].head())

# Perform sentiment analysis
def get_sentiment(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        return 'positive'
    elif blob.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['clean_text'].apply(get_sentiment)

# Display the first few rows with sentiments
print("\nDataset with Sentiments:")
print(df[[text_column, 'clean_text', 'sentiment']].head())

# Plot the distribution of sentiments
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df, order=['positive', 'neutral', 'negative'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Display word clouds for each sentiment
sentiments = ['positive', 'negative', 'neutral']
for sentiment in sentiments:
    text = ' '.join(df[df['sentiment'] == sentiment]['clean_text'])
    if text:  # Check if text is not empty
        wordcloud = WordCloud(width=800, height=400, max_words=100).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for {sentiment.capitalize()} Sentiment')
        plt.axis('off')
        plt.show()
    else:
        print(f"No text data available for {sentiment.capitalize()} sentiment")
