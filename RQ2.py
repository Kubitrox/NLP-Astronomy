'''
=================== RESEARCH QUESTION 2 ===================
"What hardware specifications are discussed and 
desired most frequently by the r/astronomy community?"
-----------------------------------------------------------
Discussed and desired - we would try to count how many times some specifications are 
discussed and check sentiment of comments about them

'''

import json
import matplotlib.pyplot as plt


telescopes = [
    "Celestron",
    "Meade",
    "Orion",
    "Zhumell",
    "Explore Scientific",
    "Vixen Optics",
    "GSO (Guan Sheng Optical)",
    "GSO",
    "iOptron",
    "William Optics",
    "Astro-Physics",
    "Takahashi",
    "Televue",
    "Lunt Solar Systems",
    "Coronado",
    "Bresser",
    "Unistellar",
    "Questar",
    "Officina Stellare",
    "SharpStar",
    "Borg",
    "TS-Optics",
    "Stellarvue",
    "Apogee Instruments",
    "Planewave Instruments",
    "10Micron",
    "SkyWatcher USA",
    "SkyWatcher",
    "Sky-watcher",
    "CFF Telescopes",
    "CFF",
    "Hubble Optics"
]

cameras = ["Sony", 
            "Canon", 
            "Nikon", 
            "Fujifilm", 
            "Panasonic", 
            "Olympus", 
            "ZWO", 
            "QHY", 
            "Atik"]


from HelperFunctions import open_preprocessed_comments_json, open_comments_entry_id, sentiment_analysis, check_features, calculate_average_compound_scores, sort_avg_compound_scores, plot_avg_compound_scores, plot_feature_mentions, turn_lowercase, delete_low_counts


processed_telescopes = turn_lowercase(telescopes)
processed_cameras = turn_lowercase(cameras)

processed_comments = open_preprocessed_comments_json()
not_processed_comments = open_comments_entry_id()

vs_id_dict = sentiment_analysis(not_processed_comments)

feature_mentions_telescopes = check_features(processed_comments, processed_telescopes)
feature_mentions_cameras = check_features(processed_comments, processed_cameras)

feature_mentions_telescopes, telescopes = delete_low_counts(feature_mentions_telescopes, telescopes)
feature_mentions_cameras, cameras = delete_low_counts(feature_mentions_cameras, cameras)

processed_telescopes = turn_lowercase(telescopes)
processed_cameras = turn_lowercase(cameras)

keyword_avg_compound_scores_telescopes = calculate_average_compound_scores(processed_comments, processed_telescopes, vs_id_dict, feature_mentions_telescopes)
keyword_avg_compound_scores_cameras = calculate_average_compound_scores(processed_comments, processed_cameras, vs_id_dict, feature_mentions_cameras)

sorted_avg_compound_scores_telescopes = sort_avg_compound_scores(keyword_avg_compound_scores_telescopes, print_scores=False)
sorted_avg_compound_scores_cameras = sort_avg_compound_scores(keyword_avg_compound_scores_cameras, print_scores=False)

plot_feature_mentions(feature_mentions_telescopes)
plot_avg_compound_scores(sorted_avg_compound_scores_telescopes)

plot_feature_mentions(feature_mentions_cameras)
plot_avg_compound_scores(sorted_avg_compound_scores_cameras)

from transformers import BertTokenizerFast, DataCollatorForTokenClassification, BertForTokenClassification, TrainingArguments, Trainer, pipeline
from datasets import Dataset

# --- Augment Dataset with Camera Companies ---
def augment_dataset_with_companies(companies, templates):
    augmented_data = []
    for company in companies:
        for template in templates:
            augmented_data.append(template.format(company))
    return augmented_data

templates_cameras = [
    "{} cameras are excellent for astrophotography.",
    "I love my {} DSLR for capturing night skies.",
    "Recently got a {} mirrorless camera and it’s amazing."
]

templates_telescopes = [
    "{} telescopes are excellent for astrophotography.",
    "I love my {} reflector for capturing deep-sky objects.",
    "Recently got a {} refractor telescope, and it’s amazing for observing planets.",
    "The {} Dobsonian is perfect for viewing faint galaxies and nebulae.",
    "I use my {} 10S telescope for astrophotography every clear night."
]

augmented_data_cameras = augment_dataset_with_companies(cameras, templates_cameras)
augmented_data_telescopes = augment_dataset_with_companies(telescopes, templates_telescopes)
print("\nAugmented data with cameras:")
print(augmented_data_cameras[:5])
print("\nAugmented data with telescopes:")
print(augmented_data_telescopes[:5])

# Sample dataset
data = {
    "tokens": [
        ["I", "love", "my", "Canon", "EOS", "Ra", "camera"],
        ["Sony", "Alpha", "is", "great", "for", "astrophotography"],
        ["The", "equipment", "used", "was", "a", "Celestron", "5SE", "and", "ZWO", "ASI294MC", ".", 
         "28,000", "frames", "stacked", "at", "35%", "on", "ASIStudio", ",", "processed", "further", "on", "Lightroom", "."],
    ],
    "tags": [
        ["O", "O", "O", "B-Camera", "I-Camera", "I-Camera", "O"],
        ["B-Camera", "I-Camera", "O", "O", "O", "O"],
        ["O", "O", "O", "O", "O", "B-Telescope", "I-Telescope", "O", "B-Camera", "I-Camera", "O", 
         "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
    ]
}

# Expanded label map to include telescopes
label_map = {
    "O": 0, 
    "B-Camera": 1, 
    "I-Camera": 2, 
    "B-Telescope": 3, 
    "I-Telescope": 4, 
}

camera_keywords = ["DSLR", "mirrorless"]
telescope_keywords = ["Dobsonian", "10S"]

def create_tokens_and_tags(augmented_sentences, companies, keywords, label1, label2):
    tokens, tags = [], []
    for sentence in augmented_sentences:
        words = sentence.split()
        sentence_tokens = []
        sentence_tags = []
        for word in words:
            sentence_tokens.append(word)
            if word in companies:
                sentence_tags.append(label1)
            elif word in keywords:
                sentence_tags.append(label2)
            else:
                sentence_tags.append("O")
        tokens.append(sentence_tokens)
        tags.append(sentence_tags)
    return tokens, tags

# Augment the training data
augmented_tokens_cameras, augmented_tags_cameras = create_tokens_and_tags(augmented_data_cameras, 
                                                                          cameras, camera_keywords, "B-Camera", "I-Camera")
augmented_tokens_telescopes, augmented_tags_telescopes = create_tokens_and_tags(augmented_data_telescopes, 
                                                                          telescopes, telescope_keywords, "B-Telescope", "I-Telescope")

# Add augmented data to the original dataset
data["tokens"].extend(augmented_tokens_cameras)
data["tags"].extend(augmented_tags_cameras)
data["tokens"].extend(augmented_tokens_telescopes)
data["tags"].extend(augmented_tags_telescopes)

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Convert dataset to Hugging Face format
dataset = Dataset.from_dict(data)

# Tokenization function with padding and truncation
def tokenize_and_align_labels(examples):
    # Tokenize inputs with padding and truncation
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=True,  # Ensures consistent lengths
        is_split_into_words=True
    )

    # Align labels with tokens
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Maps tokens to original words
        label_ids = [-100 if word_id is None else label_map[label[word_id]] for word_id in word_ids]
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize and align labels in the dataset
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Load the pre-trained model
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))

# Define data collator for token classification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator  # Automatically handles padding and batching
)

# Train the model
trainer.train()

# Use the model for inference
text = "Canon E217 is the best camera. It is much better than Sony P10"
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
ner_pipeline_output = ner_pipeline(text)

tokens = []
labels = []

for token in ner_pipeline_output:
    tokens.append(token["word"])
    labels.append(token["entity"])

print("\nExample entities from BERT:")
print(tokens)
print(labels)

filtered_sentences = []

for sentence, id in not_processed_comments:
    ner_pipeline_output = ner_pipeline(sentence)
    tokens = []
    labels = []

    for token in ner_pipeline_output:
        tokens.append(token["word"])
        labels.append(token["entity"])

    if any(label in labels for label in ["LABEL_1", "LABEL_2", "LABEL_3", "LABEL_4"]):
        filtered_sentences.append([tokens, labels])

# Initialize lists for cameras and telescopes
cameras = []
telescopes = []

# Loop through each sentence
for tokens, labels in filtered_sentences:
    current_camera = []  # To capture continuous camera tokens
    current_telescope = []  # To capture continuous telescope tokens

    for token, label in zip(tokens, labels):
        # Check for camera labels
        if label in ["LABEL_1", "LABEL_2"]:
            # Append token to current camera mention
            if token.startswith("##") and current_camera:
                current_camera[-1] += token[2:]  # Merge with the previous token
            else:
                current_camera.append(token)
        else:
            if current_camera:  # Append a complete camera mention
                cameras.append(" ".join(current_camera))
                current_camera = []

        # Check for telescope labels
        if label in ["LABEL_3", "LABEL_4"]:
            # Append token to current telescope mention
            if token.startswith("##") and current_telescope:
                current_telescope[-1] += token[2:]  # Merge with the previous token
            else:
                current_telescope.append(token)
        else:
            if current_telescope:  # Append a complete telescope mention
                telescopes.append(" ".join(current_telescope))
                current_telescope = []

    # Handle any remaining tokens
    if current_camera:
        cameras.append(" ".join(current_camera))
    if current_telescope:
        telescopes.append(" ".join(current_telescope))

processed_telescopes = turn_lowercase(telescopes)
processed_cameras = turn_lowercase(cameras)

feature_mentions_telescopes = check_features(processed_comments, processed_telescopes)
feature_mentions_cameras = check_features(processed_comments, processed_cameras)

keyword_avg_compound_scores_telescopes = calculate_average_compound_scores(processed_comments, processed_telescopes, vs_id_dict, feature_mentions_telescopes)
keyword_avg_compound_scores_cameras = calculate_average_compound_scores(processed_comments, processed_cameras, vs_id_dict, feature_mentions_cameras)

sorted_avg_compound_scores_telescopes = sort_avg_compound_scores(keyword_avg_compound_scores_telescopes, print_scores=False)
sorted_avg_compound_scores_cameras = sort_avg_compound_scores(keyword_avg_compound_scores_cameras, print_scores=False)

plot_feature_mentions(feature_mentions_telescopes)
plot_avg_compound_scores(sorted_avg_compound_scores_telescopes)

plot_feature_mentions(feature_mentions_cameras)
plot_avg_compound_scores(sorted_avg_compound_scores_cameras)
