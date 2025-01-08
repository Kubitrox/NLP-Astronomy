import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

nltk.download('punkt') # Im not sure if it has to be downloaded everytime, but I dont know how to do it otherwise
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Normalize text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
    # Remove abbreviations
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens

def preprocess_json_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            if entry['author'] != '[deleted]':
                entry['body'] = preprocess_text(entry['body'])
                data.append(entry)
    
    return data

comments_path = 'astronomy_comments.ndjson'
submissions_path = 'astronomy_submissions.ndjson'

processed_comments = preprocess_json_file(comments_path)

# The submissions do not have 'body', so the function does not work here.
# Also, we have to decide if we are going to use them anyway.
# processed_submissions = preprocess_json_file(submissions_path)

keys_to_keep = {'body', 'id', 'author', 'score', 'permalink'}

# Filter the entries to keep only the specified keys
def filter_entries(processed_data):
    for entry in processed_data:
        keys_to_remove = set(entry.keys()) - keys_to_keep
        for key in keys_to_remove:
            del entry[key]

filter_entries(processed_comments)
# filter_entries(processed_submissions)

with open('processed_astronomy_comments.ndjson', 'w') as file:
    json.dump(processed_comments, file, indent=4)

# with open('processed_astronomy_submissions.ndjson', 'w') as file:
#     json.dump(processed_submissions, file, indent=4)