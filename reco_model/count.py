import json

# Load the JSON file
with open('arxiv_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Iterate through categories and count the articles
category_counts = {category: len(articles) for category, articles in data.items()}

# Display the results
for category, count in category_counts.items():
    print(f"{category}: {count} articles")

total_categories = len(category_counts)
total_articles = sum(category_counts.values())
print(f"\nTotal categories: {total_categories}")
print(f"Total articles: {total_articles}")
