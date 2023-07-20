import os

from decouple import config
from matplotlib import pyplot as plt
from retinaface import RetinaFace

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np

import cv2
from deepface import DeepFace

import sqlite3

from celery import Celery

# setup

project_name = config("PROJECT_NAME")
# Define a Celery app
app = Celery('tasks', broker="redis://localhost:6379")
# Define the emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

#  this is the name of the project

# Define a dictionary to keep track of emotion counts
emotion_counts = {e: 0 for e in emotions}

# Define a timer to save emotion counts every 5 minutes
timer = 0
SAVE_INTERVAL = 300  # 5 minutes in seconds

# Define a variable to keep track of the most annoyed frame and other emotions
most_annoyed_frame = None
most_annoyed_score = 0
# disgust
most_disgust_frame = None
most_disgust_score = 0
# happy
most_happy_frame = None
most_happy_score = 0
# fear
most_fear_frame = None
most_fear_score = 0
# fear
most_sad_frame = None
most_sad_score = 0

# fear
most_surprise_frame = None
most_surprise_score = 0

# fear
most_neutral_frame = None
most_neutral_score = 0

# Define a variable to keep track of the most annoyed frame and other emotions
male_most_annoyed_frame = None
male_most_annoyed_score = 0
# disgust
male_most_disgust_frame = None
male_most_disgust_score = 0
# happy
male_most_happy_frame = None
male_most_happy_score = 0
# fear
male_most_fear_frame = None
male_most_fear_score = 0
# fear
male_most_sad_frame = None
male_most_sad_score = 0

# fear
male_most_surprise_frame = None
male_most_surprise_score = 0

# fear
male_most_neutral_frame = None
male_most_neutral_score = 0

# Define a variable to keep track of the most annoyed frame and other emotions
female_most_annoyed_frame = None
female_most_annoyed_score = 0
# disgust
female_most_disgust_frame = None
female_most_disgust_score = 0
# happy
female_most_happy_frame = None
female_most_happy_score = 0
# fear
female_most_fear_frame = None
female_most_fear_score = 0
# fear
female_most_sad_frame = None
female_most_sad_score = 0

# fear
female_most_surprise_frame = None
female_most_surprise_score = 0

# fear
female_most_neutral_frame = None
female_most_neutral_score = 0


@app.task
def update_dominant_database(emotion_label, current_img, emotion_score):
    """
    this function updates the dominant emotion in the database
    """
    conn = sqlite3.connect(f'{project_name}/db.sqlite')
    cursor = conn.cursor()

    directory = f'{project_name}/dominant'

    if not os.path.exists(directory):
        os.makedirs(directory)

    current_img_path = f"{directory}/{emotion_label}.jpg"
    cv2.imwrite(current_img_path, current_img)

    ### loop through and delete the user past emotion in other label

    for image in os.listdir(f"{project_name}/dominant/"):
        label_image = image.replace(".jpg", "")
        print("label image", label_image)
        if label_image != emotion_label:
            # check if the image label of the image is on other labels
            try:
                print("the images", f"{project_name}/dominant/{image}")
                result = DeepFace.verify(img1_path=current_img_path, img2_path=f"{project_name}/dominant/{image}",
                                         enforce_detection=False)
                if result.get("verified"):
                    os.remove(f"{project_name}/dominant/{image}")
                # also delete the label on the database
                cursor.execute("DELETE FROM dominant_emotions WHERE emotion_label = ?", (label_image,))
                conn.commit()
            except Exception as a:
                print("Error occurred ")

    cursor.execute(
        'CREATE TABLE IF NOT EXISTS dominant_emotions (id INTEGER PRIMARY KEY, emotion_label TEXT, dominant_frame Text'
        ',score FLOAT)')

    # Get the current dominant emotion record from the database
    cursor.execute(f"SELECT * FROM dominant_emotions WHERE emotion_label = '{emotion_label}'")
    current_record = cursor.fetchone()

    # If no record exists, insert a new record
    if current_record is None:
        cursor.execute(f"INSERT INTO dominant_emotions (emotion_label, dominant_frame, score) VALUES (?, ?, ?)",
                       (emotion_label, current_img_path, emotion_score))
    # If a record exists, update it if the new score is greater than the current score
    else:
        current_score = float(current_record[3])
        if emotion_score > current_score:
            cursor.execute(f"UPDATE dominant_emotions SET dominant_frame = ?,score = ? WHERE emotion_label = ?",
                           (emotion_label, current_img_path, emotion_score))
    conn.commit()
    conn.close()


@app.task
def update_male_dominant_database(emotion_label, current_img, emotion_score):
    """
    this function updates the dominant emotion in the database
    """
    directory = f'{project_name}/male'

    if not os.path.exists(directory):
        os.makedirs(directory)

    current_img_path = f"{directory}/{emotion_label}.jpg"
    cv2.imwrite(current_img_path, current_img)

    conn = sqlite3.connect(f'{project_name}/db.sqlite')
    cursor = conn.cursor()

    ### loop through and delete the user past emotion in other label

    for image in os.listdir(f"{project_name}/male/"):
        label_image = image.replace(".jpg", "")
        print("label image", label_image)
        if label_image != emotion_label:
            # check if the image label of the image is on other labels
            try:
                print("the images", f"{project_name}/male/{image}")
                result = DeepFace.verify(img1_path=current_img_path, img2_path=f"{project_name}/male/{image}",
                                         enforce_detection=False)
                if result.get("verified"):
                    os.remove(f"{project_name}/male/{image}")
                # also delete the label on the database
                cursor.execute("DELETE FROM male_dominant_emotions WHERE emotion_label = ?", (label_image,))
                conn.commit()
            except Exception as a:
                print("Error occurred ", a)

    cursor.execute(
        'CREATE TABLE IF NOT EXISTS male_dominant_emotions (id INTEGER PRIMARY KEY, emotion_label TEXT, dominant_frame Text'
        ',score FLOAT)')

    # Get the current dominant emotion record from the database
    cursor.execute(f"SELECT * FROM male_dominant_emotions WHERE emotion_label = '{emotion_label}'")
    current_record = cursor.fetchone()

    # If no record exists, insert a new record
    if current_record is None:
        cursor.execute(f"INSERT INTO male_dominant_emotions (emotion_label, dominant_frame, score) VALUES (?, ?, ?)",
                       (emotion_label, current_img_path, emotion_score))
    # If a record exists, update it if the new score is greater than the current score
    else:
        current_score = float(current_record[3])
        if emotion_score > current_score:
            cursor.execute(f"UPDATE male_dominant_emotions SET dominant_frame = ?,score = ? WHERE emotion_label = ?",
                           (emotion_label, current_img_path, emotion_score))
    conn.commit()
    conn.close()


@app.task
def update_female_dominant_database(emotion_label, current_img, emotion_score):
    """
    this function updates the dominant emotion in the database
    """
    directory = f'{project_name}/female'

    if not os.path.exists(directory):
        os.makedirs(directory)

    current_img_path = f"{directory}/{emotion_label}.jpg"

    cv2.imwrite(current_img_path, current_img)

    conn = sqlite3.connect(f'{project_name}/db.sqlite')
    cursor = conn.cursor()

    ### loop through and delete the user past emotion in other label

    for image in os.listdir(f"{project_name}/female/"):
        label_image = image.replace(".jpg", "")
        print("label image", label_image)
        if label_image != emotion_label:
            # check if the image label of the image is on other labels
            try:
                print("the images", f"{project_name}/female/{image}")
                result = DeepFace.verify(img1_path=current_img_path, img2_path=f"{project_name}/female/{image}",
                                         enforce_detection=False)
                if result.get("verified"):
                    os.remove(f"{project_name}/female/{image}")
                # also delete the label on the database
                cursor.execute("DELETE FROM female_dominant_emotions WHERE emotion_label = ?", (label_image,))
                conn.commit()
            except Exception as a:
                print("Error occurred ", a)

    cursor.execute(
        'CREATE TABLE IF NOT EXISTS female_dominant_emotions (id INTEGER PRIMARY KEY, emotion_label TEXT, dominant_frame Text'
        ',score FLOAT)')

    # Get the current dominant emotion record from the database
    cursor.execute(f"SELECT * FROM female_dominant_emotions WHERE emotion_label = '{emotion_label}'")
    current_record = cursor.fetchone()

    # If no record exists, insert a new record
    if current_record is None:
        cursor.execute(f"INSERT INTO female_dominant_emotions (emotion_label, dominant_frame, score) VALUES (?, ?, ?)",
                       (emotion_label, current_img_path, emotion_score))
    # If a record exists, update it if the new score is greater than the current score
    else:
        current_score = float(current_record[3])
        if emotion_score > current_score:
            cursor.execute(f"UPDATE female_dominant_emotions SET dominant_frame = ?,score = ? WHERE emotion_label = ?",
                           (emotion_label, current_img_path, emotion_score))
    conn.commit()
    conn.close()


@app.task
def generate_most_dominant_emotion(emotion_label, emotion_score, current_img):
    """
    this is used to generate the most dominant emotions among all other emotions available
    :return:
    """

    global most_annoyed_frame, most_annoyed_score, most_disgust_frame, most_disgust_score, most_happy_frame, \
        most_happy_score, most_fear_frame, most_fear_score, most_sad_frame, most_sad_score, most_surprise_frame, \
        most_surprise_score, most_neutral_frame, most_neutral_score

    # Update the most annoyed frame if applicable
    if emotion_label == 'angry' and emotion_score > most_annoyed_score:
        most_annoyed_frame = current_img
        most_annoyed_score = emotion_score
        update_dominant_database('annoyed', current_img, emotion_score)

    # Update the most disgust frame if applicable
    if emotion_label == 'disgust' and emotion_score > most_disgust_score:
        most_disgust_frame = current_img
        most_disgust_score = emotion_score
        update_dominant_database(emotion_label, current_img, emotion_score)

    # Update the most happy frame if applicable
    if emotion_label == 'happy' and emotion_score > most_happy_score:
        most_happy_frame = current_img
        most_happy_score = emotion_score
        update_dominant_database(emotion_label, current_img, emotion_score)

    # Update the most fear frame if applicable
    if emotion_label == 'fear' and emotion_score > most_fear_score:
        most_fear_frame = current_img
        most_fear_score = emotion_score
        update_dominant_database(emotion_label, current_img, emotion_score)

    # Update the most sad frame if applicable
    if emotion_label == 'sad' and emotion_score > most_sad_score:
        most_sad_frame = current_img
        most_sad_score = emotion_score
        update_dominant_database(emotion_label, current_img, emotion_score)

    # Update the most surprise frame if applicable
    if emotion_label == 'surprise' and emotion_score > most_surprise_score:
        most_surprise_frame = current_img
        most_surprise_score = emotion_score
        update_dominant_database(emotion_label, current_img, emotion_score)

    # Update the most neutral frame if applicable
    if emotion_label == 'neutral' and emotion_score > most_neutral_score:
        most_neutral_frame = current_img
        most_neutral_score = emotion_score
        update_dominant_database(emotion_label, current_img, emotion_score)

    return True


@app.task
def generate_male_most_dominant_emotion(emotion_label, emotion_score, current_img):
    """
    this is used to generate the most dominant emotions among all other emotions available
    this is meant for male
    :return:
    """

    global male_most_annoyed_frame, male_most_annoyed_score, male_most_disgust_frame, male_most_disgust_score, male_most_happy_frame, \
        male_most_happy_score, male_most_fear_frame, male_most_fear_score, male_most_sad_frame, male_most_sad_score, male_most_surprise_frame, \
        male_most_surprise_score, male_most_neutral_frame, male_most_neutral_score

    # Update the most annoyed frame if applicable
    if emotion_label == 'angry' and emotion_score > male_most_annoyed_score:
        male_most_annoyed_frame = current_img
        male_most_annoyed_score = emotion_score
        update_male_dominant_database('annoyed', current_img, emotion_score)

    # Update the male_most disgust frame if applicable
    if emotion_label == 'disgust' and emotion_score > male_most_disgust_score:
        male_most_disgust_frame = current_img
        male_most_disgust_score = emotion_score
        update_male_dominant_database(emotion_label, current_img, emotion_score)

    # Update the male_most happy frame if applicable
    if emotion_label == 'happy' and emotion_score > male_most_happy_score:
        male_most_happy_frame = current_img
        male_most_happy_score = emotion_score
        update_male_dominant_database(emotion_label, current_img, emotion_score)

    # Update the male_most fear frame if applicable
    if emotion_label == 'fear' and emotion_score > male_most_fear_score:
        male_most_fear_frame = current_img
        male_most_fear_score = emotion_score
        update_male_dominant_database(emotion_label, current_img, emotion_score)

    # Update the male_most sad frame if applicable
    if emotion_label == 'sad' and emotion_score > male_most_sad_score:
        male_most_sad_frame = current_img
        male_most_sad_score = emotion_score
        update_male_dominant_database(emotion_label, current_img, emotion_score)

    # Update the male_most surprise frame if applicable
    if emotion_label == 'surprise' and emotion_score > male_most_surprise_score:
        male_most_surprise_frame = current_img
        male_most_surprise_score = emotion_score
        update_male_dominant_database(emotion_label, current_img, emotion_score)

    # Update the male_most neutral frame if applicable
    if emotion_label == 'neutral' and emotion_score > male_most_neutral_score:
        male_most_neutral_frame = current_img
        male_most_neutral_score = emotion_score
        update_male_dominant_database(emotion_label, current_img, emotion_score)

    return True


@app.task
def generate_female_most_dominant_emotion(emotion_label, emotion_score, current_img):
    """
    this is used to generate the most dominant emotions among all other emotions available
    this is meant for male
    :return:
    """
    global female_most_annoyed_frame, female_most_annoyed_score, female_most_disgust_frame, female_most_disgust_score, female_most_happy_frame, \
        female_most_happy_score, female_most_fear_frame, female_most_fear_score, female_most_sad_frame, female_most_sad_score, female_most_surprise_frame, \
        female_most_surprise_score, female_most_neutral_frame, female_most_neutral_score

    # Update the most annoyed frame if applicable
    if emotion_label == 'angry' and emotion_score > female_most_annoyed_score:
        female_most_annoyed_frame = current_img
        female_most_annoyed_score = emotion_score
        update_female_dominant_database('annoyed', current_img, emotion_score)

    # Update the female_most disgust frame if applicable
    if emotion_label == 'disgust' and emotion_score > female_most_disgust_score:
        female_most_disgust_frame = current_img
        female_most_disgust_score = emotion_score
        update_female_dominant_database(emotion_label, current_img, emotion_score)

    # Update the female_most happy frame if applicable
    if emotion_label == 'happy' and emotion_score > female_most_happy_score:
        female_most_happy_frame = current_img
        female_most_happy_score = emotion_score
        update_female_dominant_database(emotion_label, current_img, emotion_score)

    # Update the female_most fear frame if applicable
    if emotion_label == 'fear' and emotion_score > female_most_fear_score:
        female_most_fear_frame = current_img
        female_most_fear_score = emotion_score
        update_female_dominant_database(emotion_label, current_img, emotion_score)

    # Update the female_most sad frame if applicable
    if emotion_label == 'sad' and emotion_score > female_most_sad_score:
        female_most_sad_frame = current_img
        female_most_sad_score = emotion_score
        update_female_dominant_database(emotion_label, current_img, emotion_score)

    # Update the female_most surprise frame if applicable
    if emotion_label == 'surprise' and emotion_score > female_most_surprise_score:
        female_most_surprise_frame = current_img
        female_most_surprise_score = emotion_score
        update_female_dominant_database(emotion_label, current_img, emotion_score)

    # Update the female_most neutral frame if applicable
    if emotion_label == 'neutral' and emotion_score > female_most_neutral_score:
        female_most_neutral_frame = current_img
        female_most_neutral_score = emotion_score
        update_female_dominant_database(emotion_label, current_img, emotion_score)

    return True


@app.task
def generate_emotion_pie_chart():
    # Save the emotion counts in a pie chart
    # Connect to the database
    global emotion_counts
    # create the directory if it does not exists
    if not os.path.exists(project_name):
        os.makedirs(project_name)

    # Create a list of counts in the same order as the labels
    counts = [emotion_counts[e] for e in emotions]

    values = list(emotion_counts.values())

    # Remove the label with zero value from the lists
    non_zero_labels = []
    non_zero_values = []
    for i, val in enumerate(values):
        if val > 0:
            non_zero_labels.append(emotions[i])
            non_zero_values.append(val)
    labels = non_zero_labels
    values = non_zero_values

    # Set the figure size
    plt.figure(figsize=(8, 8))
    # Create the pie chart
    plt.pie(values, labels=labels, autopct='%1.1f%%', labeldistance=1.05, textprops={'fontsize': 14})

    plt.title('Emotions Pie Chart')

    plt.savefig(f"{project_name}/chart.png")
    return True


@app.task
def generate_data(frame):
    frame = np.array(frame, dtype=np.uint8)

    # Detect faces using RetinaFace
    faces = RetinaFace.extract_faces(frame, )
    for i, face in enumerate(faces):
        # Resize the frame to a smaller size for faster processing
        dsize = (400, 400)
        # fixed the color
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        current_img = cv2.resize(face, dsize)

        # Use DeepFace to detect emotions in the frame
        try:
            print("analyzing")
            detected_faces = DeepFace.analyze(current_img, actions=('emotion', 'gender',), enforce_detection=False)
            if detected_faces is not None:
                for face in detected_faces:
                    if 'region' in face:
                        # Get the region of the face coordinate
                        emotions = face['emotion']
                        emotion_label = max(emotions, key=emotions.get)
                        emotion_score = emotions[emotion_label]
                        emotion_score = round(emotion_score, 2)
                        # Increment the emotion count
                        emotion_counts[emotion_label] += 1
                        dominant_gender = face.get("dominant_gender")
                        # Generate the chart
                        generate_emotion_pie_chart()
                        #  Only generate data if the emotion score is greater than 70
                        print("The emotion score: ",emotion_score)
                        if emotion_score > 90:
                            print("Dominant reached here")
                            # update the dominant emotion
                            generate_most_dominant_emotion(emotion_label, emotion_score, current_img)
                            if dominant_gender == "Man":
                                generate_male_most_dominant_emotion(emotion_label, emotion_score, current_img)
                            elif dominant_gender == "Woman":
                                generate_female_most_dominant_emotion(emotion_label, emotion_score,
                                                                      current_img)
        except Exception as a:
            print("the error", a)



@shared_task
def generate_data(image, report_id):
    image = np.array(image, dtype=np.uint8)

    report = Report.objects.filter(id=report_id).first()
    if not report:
        print("No report found")
        return False
    # Use MTCNN to detect faces in the image
    detector = MTCNN()
    faces = detector.detect_faces(image)

    # Use DeepFace to detect emotions in the image
    for face in faces:
        # Get the region of the face coordinate
        x, y, w, h = face['box']

        # Extract the face region from the original image with padding
        padding = 20
        padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

        # Get the new coordinates of the face region in the padded image
        x_padded, y_padded = x + padding, y + padding
        w_padded, h_padded = w + 2 * padding, h + 2 * padding

        # Extract the padded face region
        padded_face = padded_image[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]

        # Resize the padded face region to 400 by 400 pixels
        current_img = resize_cv2_image(padded_face, 400, 400)
        # show the image
        img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.show()
        try:
            print("analyzing")
            detected_faces = DeepFace.analyze(image, actions=('emotion', 'gender',), enforce_detection=True)
            """
            Deepface returns a list of emotions with the regions
            [{'emotion': {'angry': 0.0005200124033576984, 'disgust': 6.884869604224655e-11,
                                           'fear': 0.0001671884078467515, 'happy': 0.0008500899236528251,
                                           'sad': 99.99600052809718, 'surprise': 4.1343011124876384e-11,
                                           'neutral': 0.0024638100162230813}, 'dominant_emotion': 'sad',
                               'region': {'x': 0, 'y': 0, 'w': 400, 'h': 400},
                               'gender': {'Woman': 2.7019817382097244, 'Man': 97.29802012443542},
                               'dominant_gender': 'Man'}]
            """

            if detected_faces is not None:
                for detected_face in detected_faces:
                    # Get the region coordinates from the 'region' dictionary
                    # x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                    # # Extract the face region from the original image
                    # face_region = image[y:y + h, x:x + w]
                    #
                    # # Resize the extracted face region to 400 by 400 pixels
                    # current_img = cv2.resize(face_region, (400, 400))
                    # for debugging lets show the image
                    # img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
                    # plt.imshow(img_rgb)
                    # plt.show()

                    if 'region' in detected_face:
                        # Get the region of the face coordinate
                        emotions = detected_face['emotion']
                        emotion_label = max(emotions, key=emotions.get)
                        emotion_score = emotions[emotion_label]
                        emotion_score = round(emotion_score, 2)
                        # Increment the emotion count
                        if emotion_label == "angry":
                            report.angry_count += 1
                        elif emotion_label == "disgust":
                            report.disgust_count += 1
                        elif emotion_label == "fear":
                            report.fear_count += 1
                        elif emotion_label == "happy":
                            report.happy_count += 1
                        elif emotion_label == "sad":
                            report.sad_count += 1
                        elif emotion_label == "surprise":
                            report.surprise_count += 1
                        elif emotion_label == "neutral":
                            report.neutral_count += 1
                        report.save()

                        dominant_gender = detected_face.get("dominant_gender")

                        # convert the image to temporary image for django to use
                        current_img = convert_opencv_to_image(current_img)

                        #  Only generate data if the emotion score is greater than 70
                        print("The emotion score: ", emotion_score)
                        if emotion_score > 90:
                            print("Dominant reached here")
                            # update the dominant emotion
                            generate_most_dominant_emotion(emotion_label, emotion_score, current_img, report_id)
                            if dominant_gender == "Man":
                                generate_male_most_dominant_emotion(emotion_label, emotion_score, current_img,
                                                                    report_id)
                            elif dominant_gender == "Woman":
                                generate_female_most_dominant_emotion(emotion_label, emotion_score,
                                                                      current_img, report_id)
        except Exception as a:
            print(a)
        return
