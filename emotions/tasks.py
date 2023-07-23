import os

from celery import shared_task
from django.utils import timezone

from emotions.models import Report
from emotions.utils import convert_opencv_to_image

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np

import cv2
from deepface import DeepFace
from django.db import models


def verify_image_if_same(current_image, report, exclude_name):
    """
    this is used to verify the image
    current_image: the image i am checking if exists on the db
    report : the report instance
    exclude_name : the  list field name i wouldn't like to check
    """

    try:
        for field in report._meta.get_fields():
            if str(field.name).startswith("most"):
                if isinstance(report._meta.get_field(field.name), models.ImageField):
                    #  if it starts with most just continue or in exclude fields
                    if field.name not in exclude_name:

                        result = DeepFace.verify(img1_path=f"media/{current_image}",
                                                 img2_path=f"media/{getattr(report, field.name)}",
                                                 enforce_detection=False)
                        if result.get("verified"):
                            # set the value of the image to none
                            setattr(report, field.name, None)
                            report.save()
                            return True

            else:
                #  if it starts with most just continue or in exclude fields
                if field.name not in exclude_name:
                    if isinstance(report._meta.get_field(field.name), models.ImageField):
                        result = DeepFace.verify(img1_path=f"media/{current_image}",
                                                 img2_path=f"media/{getattr(report, field.name)}",
                                                 enforce_detection=False)
                        if result.get("verified"):
                            # set the value of that image to none
                            setattr(report, field.name, None)
                            report.save()
                            return True
    except Exception as a:
        print("Error occurred ", a)
    return False


@shared_task
def generate_most_dominant_emotion(emotion_label, emotion_score, current_img, report_id):
    """
    this is used to generate the most dominant emotions among all other emotions available
    :return:
    """
    report = Report.objects.filter(id=report_id).first()
    if not report:
        print("No report found")
        return False

    # Update the most annoyed image if applicable
    if emotion_label == 'angry' and emotion_score > report.most_annoyed_score:
        report.most_annoyed_image = current_img
        report.most_annoyed_score = emotion_score
        report.most_annoyed_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["most_annoyed_image"])

    # Update the most disgust image if applicable
    if emotion_label == 'disgust' and emotion_score > report.most_disgust_score:
        report.most_disgust_image = current_img
        report.most_disgust_score = emotion_score
        report.most_disgust_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["most_disgust_image"])

    # Update the most happy image if applicable
    if emotion_label == 'happy' and emotion_score > report.most_happy_score:
        report.most_happy_image = current_img
        report.most_happy_score = emotion_score
        report.most_happy_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["most_happy_image"])

    # Update the most fear image if applicable
    if emotion_label == 'fear' and emotion_score > report.most_fear_score:
        report.most_fear_image = current_img
        report.most_fear_score = emotion_score
        report.most_fear_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["most_fear_image"])

    # Update the most sad image if applicable
    if emotion_label == 'sad' and emotion_score > report.most_sad_score:
        report.most_sad_image = current_img
        report.most_sad_score = emotion_score
        report.most_sad_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["most_sad_image"])

    # Update the most surprise image if applicable
    if emotion_label == 'surprise' and emotion_score > report.most_surprise_score:
        report.most_surprise_image = current_img
        report.most_surprise_score = emotion_score
        report.most_surprise_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["most_surprise_image"])

    # Update the most neutral image if applicable
    if emotion_label == 'neutral' and emotion_score > report.most_neutral_score:
        report.most_neutral_image = current_img
        report.most_neutral_score = emotion_score
        report.most_neutral_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["most_neutral_image"])

    report.save()
    return True


@shared_task
def generate_male_most_dominant_emotion(emotion_label, emotion_score, current_img, report_id):
    """
    this is used to generate the most dominant emotions among all other emotions available
    this is meant for male
    :return:
    """
    report = Report.objects.filter(id=report_id).first()
    if not report:
        print("No report found")
        return False

    # Update the most annoyed image if applicable
    if emotion_label == 'angry' and emotion_score > report.male_most_annoyed_score:
        report.male_most_annoyed_image = current_img
        report.male_most_annoyed_score = emotion_score
        report.male_most_annoyed_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["male_most_annoyed_image"])

    # Update the report.male_most disgust image if applicable
    if emotion_label == 'disgust' and emotion_score > report.male_most_disgust_score:
        report.male_most_disgust_image = current_img
        report.male_most_disgust_score = emotion_score
        report.male_most_disgust_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["male_most_disgust_image"])

    # Update the report.male_most happy image if applicable
    if emotion_label == 'happy' and emotion_score > report.male_most_happy_score:
        report.male_most_happy_image = current_img
        report.male_most_happy_score = emotion_score
        report.male_most_happy_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["male_most_happy_image"])

    # Update the report.male_most fear image if applicable
    if emotion_label == 'fear' and emotion_score > report.male_most_fear_score:
        report.male_most_fear_image = current_img
        report.male_most_fear_score = emotion_score
        report.male_most_fear_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["male_most_fear_image"])

    # Update the report.male_most sad image if applicable
    if emotion_label == 'sad' and emotion_score > report.male_most_sad_score:
        report.male_most_sad_image = current_img
        report.male_most_sad_score = emotion_score
        report.male_most_sad_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["male_most_sad_image"])

    # Update the report.male_most surprise image if applicable
    if emotion_label == 'surprise' and emotion_score > report.male_most_surprise_score:
        report.male_most_surprise_image = current_img
        report.male_most_surprise_score = emotion_score
        report.male_most_surprise_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["male_most_surprise_image"])

    # Update the report.male_most neutral image if applicable
    if emotion_label == 'neutral' and emotion_score > report.male_most_neutral_score:
        report.male_most_neutral_image = current_img
        report.male_most_neutral_score = emotion_score
        report.male_most_neutral_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["male_most_neutral_image"])

    report.save()
    return True


@shared_task
def generate_female_most_dominant_emotion(emotion_label, emotion_score, current_img, report_id):
    """
    this is used to generate the most dominant emotions among all other emotions available
    this is meant for male
    :return:
    """
    report = Report.objects.filter(id=report_id).first()

    if not report:
        print("No report found")
        return False

    # Update the most annoyed image if applicable
    if emotion_label == 'angry' and emotion_score > report.female_most_annoyed_score:
        report.female_most_annoyed_image = current_img
        report.female_most_annoyed_score = emotion_score
        report.female_most_annoyed_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["male_most_annoyed_image"])

    # Update the report.female_most disgust image if applicable
    if emotion_label == 'disgust' and emotion_score > report.female_most_disgust_score:
        report.female_most_disgust_image = current_img
        report.female_most_disgust_score = emotion_score
        report.female_most_disgust_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["male_most_annoyed_image"])

    # Update the report.female_most happy image if applicable
    if emotion_label == 'happy' and emotion_score > report.female_most_happy_score:
        report.female_most_happy_image = current_img
        report.female_most_happy_score = emotion_score
        report.female_most_happy_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["female_most_happy_image"])

    # Update the report.female_most fear image if applicable
    if emotion_label == 'fear' and emotion_score > report.female_most_fear_score:
        report.female_most_fear_image = current_img
        report.female_most_fear_score = emotion_score
        report.female_most_fear_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["female_most_fear_image"])

    # Update the report.female_most sad image if applicable
    if emotion_label == 'sad' and emotion_score > report.female_most_sad_score:
        report.female_most_sad_image = current_img
        report.female_most_sad_score = emotion_score
        report.female_most_sad_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["female_most_sad_image"])

    # Update the report.female_most surprise image if applicable
    if emotion_label == 'surprise' and emotion_score > report.female_most_surprise_score:
        report.female_most_surprise_image = current_img
        report.female_most_surprise_score = emotion_score
        report.female_most_surprise_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["female_most_surprise_image"])

    # Update the report.female_most neutral image if applicable
    if emotion_label == 'neutral' and emotion_score > report.female_most_neutral_score:
        report.female_most_neutral_image = current_img
        report.female_most_neutral_score = emotion_score
        report.female_most_neutral_time = timezone.now()

        report.save()
        # check if the image exists
        verify_image_if_same(current_image=current_img, report=report, exclude_name=["female_most_neutral_image"])

    report.save()
    return True


@shared_task
def generate_data(image, report_id):
    image = np.array(image, dtype=np.uint8)

    report = Report.objects.filter(id=report_id).first()
    if not report:
        print("No report found")
        return False

    # Use DeepFace to detect emotions in the image
    try:
        print("analyzing")
        detected_faces = DeepFace.analyze(image, actions=('emotion', 'gender',), enforce_detection=True)
        # Deepface returns a list of emotions with the regions
        """

        detected_faces = [{'emotion': {'angry': 0.0005200124033576984, 'disgust': 6.884869604224655e-11,
                                       'sad': 0.0001671884078467515, 'happy': 0.0008500899236528251,
                                       'fear': 99.99600052809718, 'surprise': 4.1343011124876384e-11,
                                       'neutral': 0.0024638100162230813}, 'dominant_emotion': 'sad',
                           'region': {'x': 0, 'y': 0, 'w': 400, 'h': 400},
                           'gender': {'Woman': 2.7019817382097244, 'Man': 97.29802012443542},
                           'dominant_gender': 'Man'}]

        """

        if detected_faces is not None:
            for face in detected_faces:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                # Create a copy of the image to draw the rectangle and add text
                current_img = image.copy()

                if 'region' in face:
                    # Get the region of the face coordinate
                    emotions = face['emotion']
                    emotion_label = max(emotions, key=emotions.get)
                    emotion_score = emotions[emotion_label]
                    emotion_score = round(emotion_score, 2)
                    # Draw a rectangle around the face
                    cv2.rectangle(current_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Add the emotion label as text above the rectangle
                    label_text = f"{emotion_label} ({round(emotion_score, 2)})"
                    cv2.putText(current_img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                    # for debugging lets show the image
                    # img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
                    # plt.imshow(img_rgb)
                    # plt.show()
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

                    dominant_gender = face.get("dominant_gender")

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
