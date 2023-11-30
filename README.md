This is Software Engineering Project Group 6!

This program's goal is to enhance users' browsing experience when searching for articles on Arxiv. This software
uses graph neural network to predict how similar articles may be to recommend very semantically similar ones, also
providing translation and summarisation, removing the need to endlessly read the vast amount of papers in order to
find a particular one!

Main file of this program is app.py, where the main loop is.

To install the app and run, you have to download the translation model from here
https://drive.google.com/drive/u/2/folders/1ZvX11tCMK2hlaEk5yZpwArI3T7sAwNr4

Deployment on AWS cloud:
We implemented 4 EC2 Ubuntu instances for each of our application’s components: summary model, translation model, recommendation model and app. App instances can request model instances through Uvicorn server setup on each instance. Library we used for communication purposes is FastAPI. Client (App instance) sends a post and get HTTP requests to the model instances. All servers are running on the corresponding EC2 instance’s IP, port is set up independently (translation - 8000, recommendation - 8001, summary - 8002, app - Streamlit’s default, 8501). 

The additional features we used: 
Application Load Balancer for all our instances as a target group
Auto Scaling for the same target group. The group is rescaled according to the CPU average usage. Desired capacity was set at 4. Healthy percentage was set up at 0% minimum and 100% maximum for Auto Scaling to create and terminate new instances upon reaching the threshold.
No SQS or any other type of orchestration

To run the app, first the connection to all servers via SSH must be established. Then the following line must be prompted to the shell of the servers that run the models:
	uvicorn (script_name):app --host (EC2 instance IP) --port(desirable port)
Note that each port for each server is predefined in the model_requests.py file.
In the app server, the following command must be executed:
	streamlit run app.py
It will automatically start the Streamlit application on the IP of the instance and port 8501. Then the app would be easily accessible from any browser via typing in the address:                                16.170.218.226:8501 
