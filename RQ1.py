import json
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# keywords = [" star tracking ", " ephmeris generation ", " telescope control ", " image stacking ", " photometry tools ",
#             " planetarium mode ", " star charts ", " 3d ", " augmented reality ", " spectroscopy analysis ",
#             " astrometry ", " light curve ", " radio astronomy ", " satellite tracking ", " meteor shower prediction ",
#             " augmented reality ", " ar ", " vr ", " virtual reality "]

keywords = [" sky mapping ", " star identification ", " augmented reality ", " ar ", " vr ", " virtual reality ", " 3d view ",
             " time travel ", " night mode ", " satellite tracking ", " meteor shower prediction ", " light pollution map ",
             " telescope control ",  " assistance ", " observation logs ", " augmented space probes ",
             " custom overlays ", " skysafari ", " star walk ", " skyview ", " skyguide ", " stellarium ", " night sky "]
# " zoom ", " sharing "

def check_features(processed_comments, keywords, sentiment_id_dict):
    comments = [(entry['body'], entry['id']) for entry in processed_comments]

    feature_mentions = {keyword: 0 for keyword in keywords if keyword not in [" ar ", " augmented reality ", " vr ", " virtual reality "]}
    feature_mentions[" ar/augmented reality "] = 0
    feature_mentions[" vr/virtual reality "] = 0
    feature_mentions_sentiment = {keyword: {"pos_num": 0, "neg_num": 0, "neu_num": 0} for keyword in keywords if keyword not in [" ar ", " augmented reality ", " vr ", " virtual reality "]}
    feature_mentions_sentiment[" ar/augmented reality "] = {"pos_num": 0, "neg_num": 0, "neu_num": 0}
    feature_mentions_sentiment[" vr/virtual reality "] = {"pos_num": 0, "neg_num": 0, "neu_num": 0}
    
    for comment_tokens, comment_id in comments:
        comment_text = " ".join(comment_tokens)
        for keyword in keywords:
            if keyword in comment_text:
                if keyword == " ar " or keyword == " augmented reality ":
                    feature_mentions[" ar/augmented reality "] += 1
                    if sentiment_id_dict[comment_id] == "pos":
                        feature_mentions_sentiment[" ar/augmented reality "]["pos_num"] += 1
                    elif sentiment_id_dict[comment_id] == "neg":
                        feature_mentions_sentiment[" ar/augmented reality "]["neg_num"] += 1
                    else:
                        feature_mentions_sentiment[" ar/augmented reality "]["neu_num"] += 1
                elif keyword == " vr " or keyword == " virtual reality ":
                    feature_mentions[" vr/virtual reality "] += 1
                    if sentiment_id_dict[comment_id] == "pos":
                        feature_mentions_sentiment[" vr/virtual reality "]["pos_num"] += 1
                    elif sentiment_id_dict[comment_id] == "neg":
                        feature_mentions_sentiment[" vr/virtual reality "]["neg_num"] += 1
                    else:
                        feature_mentions_sentiment[" vr/virtual reality "]["neu_num"] += 1
                else:
                    feature_mentions[keyword] += 1
                    if sentiment_id_dict[comment_id] == "pos":
                        feature_mentions_sentiment[keyword]["pos_num"] += 1
                    elif sentiment_id_dict[comment_id] == "neg":
                        feature_mentions_sentiment[keyword]["neg_num"] += 1
                    else:
                        feature_mentions_sentiment[keyword]["neu_num"] += 1


    return feature_mentions, feature_mentions_sentiment

with open('processed_astronomy_comments.ndjson', 'r', encoding='utf-8') as file:
    processed_comments = json.load(file)

not_processed_comments = []
with open('astronomy_comments.ndjson', 'r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line)
        if entry['author'] != '[deleted]':
            not_processed_comments.append((entry['body'], entry['id']))


sentiment_id_dict = {}
analyzer = SentimentIntensityAnalyzer()
for sentence, id in not_processed_comments:
    vs = analyzer.polarity_scores(sentence)
    sentiment = max({k: v for k, v in vs.items() if k != 'compound'}, key=vs.get)
    sentiment_id_dict[id] = sentiment

feature_mentions, feature_mentions_sentiment = check_features(processed_comments, keywords, sentiment_id_dict)

# for feature, count in feature_mentions.items():
#     print(f"{feature}: {count} mentions")

for feature, sentiment in feature_mentions_sentiment.items():
    print(f"{feature}: {sentiment}")

# features = list(feature_mentions.keys())
# counts = list(feature_mentions.values())

# plt.figure(figsize=(10, 6))
# plt.bar(features, counts, color='skyblue')
# plt.ylabel('Number of Mentions')
# plt.xlabel('Features')
# plt.title('Feature Mentions in Astronomy Comments')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()