import pandas as pd
import json
import re
import tldextract


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
