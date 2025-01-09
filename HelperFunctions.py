import pandas as pd
import json
import re
import tldextract
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def load_ndjson(file_path):

    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    return pd.DataFrame(data)


def load_reddit_data(comments_df, submissions_df):

    comments = comments_df.to_dict('list')
    submissions = submissions_df.to_dict('list')

    return comments, submissions


def search_pattern(pattern, comments):
    matches = []
    for comment in comments:
        if re.search(pattern, comment):
            matches.append(comment)

    return matches


def extract_urls(comments):

    regex = re.compile(r'https?://\S+|www\.\S+')
    urls = []

    for comment in comments:
        matches = re.findall(regex, comment)
        urls.extend(matches)

    return urls


def extract_domain(url):
    return tldextract.extract(url).registered_domain


def save_comments_to_json(data, output_dir):

    d = [{'id': id, 'author': author, 'link_id': link_id, 'parent_id': parent_id, 'created_utc': created_utc, 'body': body, 'score': score, 'sentences': sentences, 'words': words} for id, author, link_id, parent_id, created_utc, body, score, sentences, words in zip(data['id'], data['author'], data['link_id'], data['parent_id'], data['created_utc'], data['body'], data['score'], data['sentences'], data['words'])]

    with open(output_dir, "w") as output_file:
        json.dump(d, output_file, indent=4)


def save_submissions_to_json(data, output_dir):

    d = [{'title': title, 'selftext': selftext, 'created_utc': created_utc, 'url': url, 'score': score, 'num_comments': num_comments, 'sentences': sentences, 'words': words} for title, selftext, created_utc, url, score, num_comments, sentences, words in zip(data['title'], data['selftext'], data['created_utc'], data['url'], data['score'], data['num_comments'], data['sentences'], data['words'])]

    with open(output_dir, "w") as output_file:
        json.dump(d, output_file, indent=4)


def write_to_text_file(data, output_dir):

    with open(output_dir, 'w') as file:
        file.writelines(line + '\n' for line in data)


def check_features(processed_comments, keywords):
    comments = [entry['words'] for entry in processed_comments]

    feature_mentions = {keyword: 0 for keyword in keywords if keyword not in [" ar ", " augmented reality ", " vr ", " virtual reality "]}
    feature_mentions[" ar/augmented reality "] = 0
    feature_mentions[" vr/virtual reality "] = 0
    
    for comment_tokens in comments:
        comment_text = " ".join(comment_tokens)
        for keyword in keywords:
            if keyword in comment_text:
                if keyword == " ar " or keyword == " augmented reality ":
                    feature_mentions[" ar/augmented reality "] += 1
                elif keyword == " vr " or keyword == " virtual reality ":
                    feature_mentions[" vr/virtual reality "] += 1
                else:
                    feature_mentions[keyword] += 1

    return feature_mentions

def calculate_average_compound_scores(processed_comments, keywords, sentiment_scores, feature_mentions):
    comments = [(entry['words'], entry['id']) for entry in processed_comments]

    keyword_compound_scores = {keyword: 0 for keyword in keywords if keyword not in [" ar ", " augmented reality ", " vr ", " virtual reality "]}
    keyword_compound_scores[" ar/augmented reality "] = 0
    keyword_compound_scores[" vr/virtual reality "] = 0

    for comment_tokens, comment_id in comments:
        comment_text = " ".join(comment_tokens)
        for keyword in keywords:
            if keyword in comment_text:
                compound_score = sentiment_scores[comment_id]["compound"]
                if keyword == " ar " or keyword == " augmented reality ":
                    keyword_compound_scores[" ar/augmented reality "] += compound_score
                elif keyword == " vr " or keyword == " virtual reality ":
                    keyword_compound_scores[" vr/virtual reality "] += compound_score
                else:
                    keyword_compound_scores[keyword] += compound_score

    keyword_avg_compound_scores = {}
    for keyword, total_score in keyword_compound_scores.items():
        mentions = feature_mentions.get(keyword, 0)
        keyword_avg_compound_scores[keyword] = total_score / mentions if mentions > 0 else 0

    return keyword_avg_compound_scores


def open_preprocessed_comments_json():
    with open('preprocessed_comments_json_format.ndjson', 'r', encoding='utf-8') as file:
        processed_comments = json.load(file)

    return processed_comments


def open_comments_entry_id():
    not_processed_comments = []
    with open('astronomy_comments.ndjson', 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            not_processed_comments.append((entry['body'], entry['id']))

    return not_processed_comments

def sentiment_analysis(not_processed_comments):
    vs_id_dict = {}
    analyzer = SentimentIntensityAnalyzer()
    for sentence, id in not_processed_comments:
        vs = analyzer.polarity_scores(sentence)
        vs_id_dict[id] = vs

    return vs_id_dict

def sort_avg_compound_scores(keyword_avg_compound_scores, print_scores=False):
    sorted_avg_scores = sorted(keyword_avg_compound_scores.items(), key=lambda item: item[1], reverse=True)

    if print_scores:
        for feature, avg_score in sorted_avg_scores:
            print(f"{feature}: Average Compound Score = {avg_score}")
    
    return sorted_avg_scores

def plot_avg_compound_scores(sorted_avg_scores):
    features = [item[0] for item in sorted_avg_scores]
    avg_scores = [item[1] for item in sorted_avg_scores]

    plt.figure(figsize=(10, 6))
    plt.bar(features, avg_scores, color='skyblue')
    plt.ylabel('Average Compound Score')
    plt.xlabel('Features')
    plt.title('Average Compound Scores for Features in Astronomy Comments')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_feature_mentions(feature_mentions):
    features = list(feature_mentions.keys())
    counts = list(feature_mentions.values())

    plt.figure(figsize=(10, 6))
    plt.bar(features, counts, color='skyblue')
    plt.ylabel('Number of Mentions')
    plt.xlabel('Features')
    plt.title('Feature Mentions in Astronomy Comments')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()