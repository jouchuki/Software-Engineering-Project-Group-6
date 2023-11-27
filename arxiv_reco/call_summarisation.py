from transformers import pipeline
import requests
import PyPDF2
import io
from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()


def get_summary(link, streamlit_mode=True):
    if streamlit_mode is False:
        answer = input("Do you want to get an AI-generated summary of the full article?(y/n)")
        if answer == "y":
            summarizer = pipeline(
                task="summarization",
                model="t5-small",
                min_length=20,
                max_length=150,
                truncation=True,
                model_kwargs={"cache_dir": '/Documents/Huggin_Face/'},
            )

            # Download the PDF
            arxiv_id = link.split("/")[-1]

            # Construct the PDF URL
            pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'

            # Download the PDF file
            response = requests.get(pdf_url)

            if response.status_code == 200:
                # Create a BytesIO object from the downloaded content
                pdf_bytes = io.BytesIO(response.content)

                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(pdf_bytes)

                # Extract text from each page
                text = ''
                for page_number in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_number].extract_text()

                processed_text = text.replace('\n', ' ')
                summary = summarizer(processed_text, max_length=250, min_length=15, do_sample=False)[0]['summary_text']
                return summary
            else:
                print(f"Failed to download PDF from {pdf_url}. Status code: {response.status_code}")
                return None
    if streamlit_mode is True:
        summarizer = pipeline(
            task="summarization",
            model="t5-small",
            min_length=20,
            max_length=150,
            truncation=True,
            model_kwargs={"cache_dir": '/Documents/Huggin_Face/'},
        )

        # Download the PDF
        arxiv_id = link.split("/")[-1]

        # Construct the PDF URL
        pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'

        # Download the PDF file
        response = requests.get(pdf_url)

        if response.status_code == 200:
            # Create a BytesIO object from the downloaded content
            pdf_bytes = io.BytesIO(response.content)

            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_bytes)

            # Extract text from each page
            text = ''
            for page_number in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_number].extract_text()

            processed_text = text.replace('\n', ' ')
            summary = summarizer(processed_text, max_length=250, min_length=15, do_sample=False)[0]['summary_text']
            return summary
        else:
            print(f"Failed to download PDF from {pdf_url}. Status code: {response.status_code}")
            return None

@app.post("/summary/")
async def summary_api(link: str):
    try:
        return {"summary text": get_summary(link)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))