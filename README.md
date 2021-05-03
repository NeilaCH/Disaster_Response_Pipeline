# Disaster Response Pipeline
## _Udacity Nanodegree Program_
The project is for the [Data Scientist Udacity Nanodegree program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). It consists of wroking on a data set containing real messages that were sent during disaster events. The main target is to build a machine learning pipeline to categorize these events in order to send the messages to an appropriate disaster relief agency.
## Before running the project
Before running the project you must have the following pakacges:
- Python 3.7
- Pandas
- scikit-learn
- Numpy
- SciPy
- SQLalchemy
- NLTK
- Punket
- Wordnet
- Flask
- Plotly
## Project Structure
The project is based on the following structure:
- ETL Pipeline Preparation.ipynb. is a python program built using jupyter notebook. It includes the code used to prepare the pipeline.
- ML Pipeline Preparation.ipynb. is a python program built using jupyter notebook. It includes the code for machine learning pipeline preparation.
- models/train_classifier.py. is a python program including machine learning pipeline scripts to train classifier and save it.
- data/process_data.py.is a python program to clean and store the ETL pipeline in database.
- data/DisasterResponse.db. is a database generated through the process_data.py, it includes messages and categories.
- data/disaster_categories.csv. is a dataset conatins the categories of messages
- data/disaster_messages.csv. is a dataset representing the messages
- app/run.py. is a flask file serves to run the web application.
- app/templates/. includes the templates of the web application as master.html and go.html

### Instructions to Run the Project
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stors in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Get your environment ids with env|grep WORK 
   Change the space-id and space-domain in the https string like this https://SPACEID-3001.SPACEDOMAIN then open  your browser and copy the link.
   In my case I used https://view6914b2f4-3001.udacity-student-workspaces.com/

### License
* The dataset used in this porject is provided by Figure Eight. Thank you for giving me this opportunity to explore a new domain
* The porject is provided by [Udacity](https://www.udacity.com). Thank you for this challenge and the support.
