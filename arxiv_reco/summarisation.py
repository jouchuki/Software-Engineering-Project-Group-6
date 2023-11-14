from transformers import pipeline
import requests
import PyPDF2
import io


def get_summary(link):
    answer = input("Do you want to get an AI-generated summary of the full article?(y/n)")
    if answer == "y":
        summarizer = pipeline(
            task="summarization",
            model="t5-small",
            min_length=20,
            max_length=40,
            truncation=True,
            model_kwargs={"cache_dir": '/Documents/Huggin_Face/'},
        )

        # Download the PDF
        response = requests.get(link)
        file_stream = io.BytesIO(response.content)

        try:
            pdf_reader = PyPDF2.PdfReader(file_stream)
            # Your code to process the PDF
        except PyPDF2.errors.PdfReadError as e:
            print("Error reading PDF file:", e)
            # Additional error handling or alternative actions

        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'

        # Basic Preprocessing (example: removing newline characters)
        processed_text = text.replace('\n', ' ')

        # Generate the summary
        summary = summarizer(processed_text, max_length=250, min_length=15, do_sample=False)[0]['summary_text']

        # Print the generated summary
        print("Summary:", summary)
        return summary
