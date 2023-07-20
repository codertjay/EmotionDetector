This is an Emotion Detector Project that uses Deep Face model which can be used in a live class
in which student and be on camera and we are able to get the report base on how the student are currently
feeling in the class to setup

## Activate Virtualenv

- Install virtualenv
  `pip install virtualenv`
- Activate virtualenv
  . (`virtualenv venv`)
  . on windows (`venv\Scripts\activate`)
  . on mac (`source venv\bin\activate`)

- Install requirements
  `pip install -r requirements.txt`

### Run Project

Create three new terminal one for project, one for task and the last for report

#### For main Project

. Before running this make sure you add the project name in the `.env` file

- `python main.py`

#### For Task

- `celery -A tasks worker -l info --pool=solo --concurrency=8`

#### For Report

- `python .\flask_app.py `

Note: Make sure you have redis installed and also the server running

### Tools Used

- opencv
- Deepface
- Retina-face
- Haar_cascade