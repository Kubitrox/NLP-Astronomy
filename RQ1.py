import json
import matplotlib.pyplot as plt


keywords = ["star tracking", "ephmeris generation", "telescope control", "image stacking", "photometry tools",
            "planetarium mode", "star charts", "3D models", "augmented reality", "spectroscopy analysis",
            "astrometry", "light curve", "radio astronomy", "satellite tracking", "meteor shower prediction"]

def check_features(comments, keywords):
    feature_mentions = {keyword: 0 for keyword in keywords}
    
    for comment_tokens in comments:
        comment_text = " ".join(comment_tokens)
        for keyword in keywords:
            if keyword in comment_text:
                feature_mentions[keyword] += 1
    
    return feature_mentions

with open('processed_astronomy_comments.ndjson', 'r', encoding='utf-8') as file:
    processed_comments = json.load(file)

comments = [entry['body'] for entry in processed_comments]

feature_mentions = check_features(comments, keywords)

for feature, count in feature_mentions.items():
    print(f"{feature}: {count} mentions")

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