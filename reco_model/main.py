import requests
import sdofgjdsj
import time

BASE_URL = 'http://export.arxiv.org/api/query?'

# Given the categories you provided
categories = [
    'astro-ph', 'cond-mat', 'gr-qc', 'hep-ex', 'hep-lat', 'hep-ph',
    'hep-th', 'math-ph', 'nlin', 'nucl-ex', 'nucl-th', 'physics',
    'quant-ph', 'math', 'CoRR', 'q-bio', 'q-fin', 'stat', 'eess', 'econ'
]


def parse_arxiv_response(xml_text):
    # Use BeautifulSoup or any XML parser to extract the metadata
    # This is based on previous discussions, adapt as needed
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(xml_text, 'xml')
    entries = soup.find_all('entry')
    data = []
    for entry in entries:
        title = entry.find('title').text
        summary = entry.find('summary').text
        authors = [author.find('name').text for author in entry.find_all('author')]
        data.append({'title': title, 'summary': summary, 'authors': authors})
    return data


all_data = {}

for category in categories:
    query = f"{BASE_URL}search_query=cat:{category}&start=0&max_results=100"
    response = requests.get(query)
    if response.status_code == 200:
        parsed_data = parse_arxiv_response(response.text)
        all_data[category] = parsed_data
    else:
        print(f"Failed to fetch data for category {category}. Status code: {response.status_code}")

    # Respect arXiv's rate limit: wait for 3 seconds before next request
    time.sleep(10)

# Save to a JSON file
with open('arxiv_data.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)


print("Data fetching complete.")
