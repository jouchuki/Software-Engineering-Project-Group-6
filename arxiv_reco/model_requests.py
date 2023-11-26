import requests
import subprocess


def start_uvicorn_server(port, app):
    cmd = f"uvicorn {app}:app --reload --port {port}"
    subprocess.Popen(cmd, shell=True)


def call_translate_api(input_text):
    # URL of the FastAPI endpoint with query parameter
    url = f'http://127.0.0.1:8000/translate/?input_text={input_text}'

    try:
        # Make a POST request to the FastAPI server
        response = requests.post(url)  # Using POST here

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response JSON and get translated text
            translated_text = response.json().get('translated_text', None)
            if translated_text is not None:
                return translated_text
            else:
                print("Translated text not found in the response.")
                return None
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def call_get_data_api(keywords):
    # URL of the FastAPI endpoint
    url = f'http://127.0.0.1:8001/get-data/?keywords={keywords}'

    # Prepare the data to be sent in the POST request
    data = {"keywords": keywords}

    try:
        # Make a POST request to the FastAPI server
        response = requests.post(url, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response JSON to get the graph data
            json_data = response.json()
            return json_data
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def call_summary_api(pdf_link):
    # URL of the FastAPI endpoint
    url = f'http://127.0.0.1:8002/summary/?link={pdf_link}'

    # Prepare the data to be sent in the POST request
    data = {"link": pdf_link}

    try:
        # Make a POST request to the FastAPI server
        response = requests.post(url, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response JSON to get the summary text
            summary_text = response.json().get('summary text', None)
            if summary_text is not None:
                return summary_text
            else:
                print("Summary text not found in the response.")
                return None
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def call_recommend_api(graph_data, article_index, num_recommendations=10):
    url = 'http://127.0.0.1:8001/recommend/'

    data = {
        "request": {
            "article_index": article_index,
            "num_recommendations": num_recommendations
        },
        "graph_data": graph_data
    }

    try:
        # Make a POST request to the FastAPI server
        response = requests.post(url, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response JSON to get the recommended indices
            recommended_indices = response.json().get('recommended_indices', None)
            return recommended_indices
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Example usage
if __name__ == "__main__":
    start_uvicorn_server(8000, "translate_api")
    start_uvicorn_server(8001, "arxiv_reco_api")
    start_uvicorn_server(8002, "call_summarisation")
    pdf_link = "https://arxiv.org/pdf/2311.13546"  # Replace with your actual PDF link
    summary_text = call_summary_api(pdf_link)
    if summary_text:
        print("Summary Text Retrieved Successfully:")
        print(summary_text)
    else:
        print("Failed to retrieve summary text.")
    keywords = "gnn"  # Replace with your actual keywords
    json_data = call_get_data_api(keywords)
    if json_data:
        print("Combined Data Retrieved Successfully.")
        print(json_data['articles'], json_data['graph'])
    else:
        print("Failed to retrieve graph data.")
    input_text = "In the first place of European Union"
    translated_text = call_translate_api(input_text)
    if translated_text:
        print(f"Translated Text: {translated_text}")
    else:
        print("No translation was returned.")
    article_index = 5
    if json_data:
        graph_data = json_data['graph']
        recommendations = call_recommend_api(graph_data, article_index, 10)
        if recommendations is not None:
            print("Recommended article indices:", recommendations)
        else:
            print("Failed to get recommendations.")