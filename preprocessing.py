import json
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
import string
import emoji
import contractions

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def retain_relevant_keys(comments_df, submissions_df):

    comments_df = comments_df[['id', 'author', 'link_id', 'parent_id', 'created_utc', 'body', 'score']].copy()
    submissions_df = submissions_df[['title', 'selftext', 'created_utc', 'url', 'score', 'num_comments']].copy()

    print("Sanity Check")
    print("Relevant keys for Comments: ", comments_df.keys())
    print("Relevant keys for Submissions: ", submissions_df.keys())

    return comments_df, submissions_df


def remove_deleted_text_fields(comments_df, submissions_df):

    comments_df = comments_df[~comments_df['body'].isin(['[deleted]', '[removed]', ''])]
    submissions_df = submissions_df[~submissions_df['selftext'].isin(['[deleted]', '[removed]', ''])]

    print("Sanity Check")

    count = 0
    for index, comment in comments_df.iterrows():
        if comment['body'] in {'[deleted]', '[removed]', ''}:
            count += 1
    print(f"Number of deleted comments after removal: {count}")

    count = 0
    for index, submission in submissions_df.iterrows():
        if submission['selftext'] in {'[deleted]', '[removed]', ''}:
            count += 1
    print(f"Number of deleted submissions after removal: {count}")

    print(f"Number of comments after removal: {comments_df.shape[0]}")
    print(f"Number of submissions after removal: {submissions_df.shape[0]}")

    return comments_df, submissions_df


def convert_timestamps_to_datetime(comments_df, submissions_df):

    comments_df['created_utc'] = pd.to_datetime(comments_df['created_utc'], unit='s')
    submissions_df['created_utc'] = pd.to_datetime(submissions_df['created_utc'], unit='s')

    return comments_df, submissions_df


def preprocess_text(text):

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Expand contractions
    text = contractions.fix(text)

    # Case Folding
    text = text.lower()

    # Remove any links
    url_pattern = r'http\S+|www\S+|https\S+'
    text = re.sub(url_pattern, '', text, flags=re.MULTILINE)

    sentences = sent_tokenize(text)

    all_sentences = []
    all_words = []

    for sentence in sentences:
        # Case folding
        sentence = sentence.lower()

        # Remove punctuation
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence)

        # Remove extra whitespaces
        sentence = re.sub(r'\s+', ' ', sentence)

        # Tokenize into words
        words = word_tokenize(sentence)

        # Remove stopwords and lemmatize
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

        # Add to lists
        if sentence != '':
            all_sentences.append(sentence)

        all_words.extend(words)

    return all_sentences, all_words
