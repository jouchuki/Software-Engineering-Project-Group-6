import os
import subprocess

def run_streamlit_app():
    # Define the directory path and the app path
    directory_path = "C:\\Users\\vsoko\\PycharmProjects\\SEProject\\Software-Engineering-Project-Group-6\\arxiv_reco"
    app_path = os.path.join(directory_path, "app.py")

    # Change to the specified directory
    os.chdir(directory_path)

    # Run the Streamlit app
    subprocess.run(["streamlit", "run", app_path])

if __name__ == "__main__":
    run_streamlit_app()
