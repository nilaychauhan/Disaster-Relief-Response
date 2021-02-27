# Disaster-Relief-Response
## Problem Statement
In this project, I've build a model to classify messages that are sent during disasters. There are 36 pre-defined categories, and examples of these categories include Aid Related, Medical Help, Search And Rescue, etc. By classifying these messages, we can allow these messages to be sent to the appropriate disaster relief agency. This project will involve the building of a basic ETL and Machine Learning pipeline to facilitate the task. This is also a multi-label classification task, since a message can belong to one or more categories.
## Data Overview
In this Project, we'll use a data set containing real messages that were sent during disaster events, which we will be using to create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency. This dataset is provided by [Figure Eight](https://appen.com/)
## File Structure
~~~~~~~
        disaster_response_pipeline
          |-- application.py
          |-- templates
                |-- go.html
                |-- master.html
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- messages_response.csv
                |-- process_data.py
          |-- models
                |-- classifier.joblib
                |-- train_classifier.py
          |-- requirements.txt
          |-- README
~~~~~~~
## Deployment
I have deployed this project using Amazon Web Services(AWS), it is live [here](http://ec2-54-146-222-127.compute-1.amazonaws.com/)
