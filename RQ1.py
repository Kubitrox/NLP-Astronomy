import json
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



keywords = [" sky mapping ", " star identification ", " augmented reality ", " ar ", " vr ", " virtual reality ", " 3d view ",
             " time travel ", " night mode ", " satellite tracking ", " meteor shower prediction ", " light pollution map ",
             " telescope control ",  " assistance ", " observation logs ", " augmented space probes ",
             " custom overlays ", " skysafari ", " star walk ", " skyview ", " skyguide ", " stellarium ", " night sky "]

def check_features(processed_comments, keywords):
    comments = [entry['body'] for entry in processed_comments]

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
    comments = [(entry['body'], entry['id']) for entry in processed_comments]

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


with open('processed_astronomy_comments.ndjson', 'r', encoding='utf-8') as file:
    processed_comments = json.load(file)

not_processed_comments = []
with open('astronomy_comments.ndjson', 'r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line)
        if entry['author'] != '[deleted]':
            not_processed_comments.append((entry['body'], entry['id']))


vs_id_dict = {}
analyzer = SentimentIntensityAnalyzer()
for sentence, id in not_processed_comments:
   vs = analyzer.polarity_scores(sentence)
   vs_id_dict[id] = vs

feature_mentions = check_features(processed_comments, keywords)
keyword_avg_compound_scores = calculate_average_compound_scores(processed_comments, keywords, vs_id_dict, feature_mentions)
sorted_avg_scores = sorted(keyword_avg_compound_scores.items(), key=lambda item: item[1], reverse=True)

for feature, avg_score in sorted_avg_scores:
    print(f"{feature}: Average Compound Score = {avg_score}")

# Plot the sorted average compound scores
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