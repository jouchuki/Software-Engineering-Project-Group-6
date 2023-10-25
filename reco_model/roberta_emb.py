import json
from transformers import RobertaTokenizer, RobertaModel
import torch

# Load the preprocessed data
with open('arxiv_data_preprocessed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Load pre-trained RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')


def get_roberta_embedding(text, model, tokenizer):
    # Tokenize the text and convert it to tensor format
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract [CLS] token's embedding and return
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy().tolist()


# Update the data with new RoBERTa embeddings
for category, articles in data.items():
    for article in articles:
        article['title_embedding'] = get_roberta_embedding(article['title'], model, tokenizer)
        article['summary_embedding'] = get_roberta_embedding(article['summary'], model, tokenizer)

# Save the updated data with RoBERTa embeddings
with open('arxiv_data_preprocessed_roberta.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
