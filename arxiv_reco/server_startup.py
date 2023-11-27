import subprocess

def start_uvicorn_server(port, app):
    cmd = f"uvicorn {app}:app --reload --port {port}"
    subprocess.Popen(cmd, shell=True)


start_uvicorn_server(8000, "translate_api")
start_uvicorn_server(8001, "arxiv_reco_api")
start_uvicorn_server(8002, "call_summarisation")
