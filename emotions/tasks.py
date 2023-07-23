import os

from celery import shared_task
from django.utils import timezone

from emotions.models import Report
from emotions.utils import convert_opencv_to_image
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np

import cv2
from deepface import DeepFace
from django.db import models


def delete_and_verify_image(report, field_name, clipped_image):
    """
    report: the report instance
    field_name : the image field name
    clipped_image : the image that was passed from webrtc that has been clippped
    this is used to delete and verify the image if it erxist before it deletes the old once
    :return:
    """
    try:
        field_start_name = f"{field_name}".replace("_image", "")
        #  the region of the image on the db
        region_x = getattr(report, f"{field_start_name}_face_x")
        region_y = getattr(report, f"{field_start_name}_face_y")
        region_width = getattr(report, f"{field_start_name}_face_width")
        region_height = getattr(report, f"{field_start_name}_face_height")

        #  get the image two path
        # this is a clipped image from open cv
        img1_path = clipped_image
        # get the path first
        img2_path = f"media/{getattr(report, field_name)}"
        read_image = cv2.imread(img2_path)
        # crop the image
        cropped_image = read_image[region_y:region_y + region_height, region_x:region_x + region_width]

        result = DeepFace.verify(img1_path=clipped_image,
                                 img2_path=cropped_image,
                                 enforce_detection=False)
        print("Verifying image")
        if result.get("verified"):
            print("Image verified")
            # set the value of the image to none
            setattr(report, field_name, None)
            report.save()
            # delete the image
            os.remove(img2_path)
            return True
    except Exception as a:
        print("Error verifying the  image :", a)
        return False


def verify_image_if_same(report, exclude_name, clipped_image):
    """
    this is used to verify the image
    current_image: the image i am checking if exists on the db
    report : the report instance
    exclude_name : the  list field name i wouldn't like to check
    clipped_image: the image cropped currently on realtime
    """
    #  i added chart_image to thr excluded filed since its not part of the faces gotten from webrtc
    # could be  male, female or most
    field_gender = exclude_name[0].split("_")[0]
    exclude_name += ["chart_image"]
    try:
        for field in report._meta.get_fields():
            #  if it starts with most just continue or in exclude fields
            if field.name in exclude_name:
                continue
            # if the field value is none just move to the next
            if str(getattr(report, field.name)) == "":
                continue
            if isinstance(report._meta.get_field(field.name), models.ImageField):
                if str(field.name).startswith(field_gender):
                    # delete the images that already exists before
                    delete_and_verify_image(report, field.name, clipped_image)
                elif str(field.name).startswith(field_gender):
                    # delete the images that already exists before
                    delete_and_verify_image(report, field.name, clipped_image)
                elif str(field.name).startswith(field_gender):
                    # delete the images that already exists before
                    delete_and_verify_image(report, field.name, clipped_image)
            tf.compat.v1.keras.backend.clear_session()
    except Exception as a:
        print("Error occurred ", a)
        tf.compat.v1.keras.backend.clear_session()
    return False


@shared_task
def generate_most_dominant_emotion(emotion_label, emotion_score, current_img, report_id, region, clipped_image):
    """
    this is used to generate the most dominant emotions among all other emotions available
    :return:
    """
    report = Report.objects.filter(id=report_id).first()
    if not report:
        print("No report found")
        return False

    # Update the most angry image if applicable
    if emotion_label == 'angry':
        if emotion_score > report.most_angry_score or str(report.most_angry_image) == "":
            report.most_angry_image = current_img
            report.most_angry_score = emotion_score
            report.most_angry_time = timezone.now()
            # save the regions also
            report.most_angry_face_x = region["x"]
            report.most_angry_face_y = region["y"]
            report.most_angry_face_width = region["w"]
            report.most_angry_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["most_angry_image"],
                                 clipped_image=clipped_image)

    # Update the most disgust image if applicable
    if emotion_label == 'disgust':
        if emotion_score > report.most_disgust_score or str(report.most_disgust_image) == "":
            report.most_disgust_image = current_img
            report.most_disgust_score = emotion_score
            report.most_disgust_time = timezone.now()
            # save the regions also
            report.most_disgust_face_x = region["x"]
            report.most_disgust_face_y = region["y"]
            report.most_disgust_face_width = region["w"]
            report.most_disgust_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["most_disgust_image"],
                                 clipped_image=clipped_image)

    # Update the most happy image if applicable
    if emotion_label == 'happy':
        if emotion_score > report.most_happy_score or str(report.most_happy_image) == "":
            report.most_happy_image = current_img
            report.most_happy_score = emotion_score
            report.most_happy_time = timezone.now()
            # save the regions also
            report.most_happy_face_x = region["x"]
            report.most_happy_face_y = region["y"]
            report.most_happy_face_width = region["w"]
            report.most_happy_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["most_happy_image"],
                                 clipped_image=clipped_image)

    # Update the most fear image if applicable
    if emotion_label == 'fear':
        if emotion_score > report.most_fear_score or str(report.most_fear_image) == "":
            report.most_fear_image = current_img
            report.most_fear_score = emotion_score
            report.most_fear_time = timezone.now()
            # save the regions also
            report.most_fear_face_x = region["x"]
            report.most_fear_face_y = region["y"]
            report.most_fear_face_width = region["w"]
            report.most_fear_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["most_fear_image"],
                                 clipped_image=clipped_image)

    # Update the most sad image if applicable
    if emotion_label == 'sad':
        if emotion_score > report.most_sad_score or str(report.most_sad_image) == "":
            report.most_sad_image = current_img
            report.most_sad_score = emotion_score
            report.most_sad_time = timezone.now()
            # save the regions also
            report.most_sad_face_x = region["x"]
            report.most_sad_face_y = region["y"]
            report.most_sad_face_width = region["w"]
            report.most_sad_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["most_sad_image"],
                                 clipped_image=clipped_image)

    # Update the most surprise image if applicable
    if emotion_label == 'surprise':
        if emotion_score > report.most_surprise_score or str(
                report.most_surprise_image) == "":
            report.most_surprise_image = current_img
            report.most_surprise_score = emotion_score
            report.most_surprise_time = timezone.now()
            # save the regions also
            report.most_surprise_face_x = region["x"]
            report.most_surprise_face_y = region["y"]
            report.most_surprise_face_width = region["w"]
            report.most_surprise_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["most_surprise_image"],
                                 clipped_image=clipped_image)

    # Update the most neutral image if applicable
    if emotion_label == 'neutral':
        if emotion_score > report.most_neutral_score or str(report.most_neutral_image) == "":
            report.most_neutral_image = current_img
            report.most_neutral_score = emotion_score
            report.most_neutral_time = timezone.now()
            # save the regions also
            report.most_neutral_face_x = region["x"]
            report.most_neutral_face_y = region["y"]
            report.most_neutral_face_width = region["w"]
            report.most_neutral_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["most_neutral_image"],
                                 clipped_image=clipped_image)

    report.save()
    return True


@shared_task
def generate_male_most_dominant_emotion(emotion_label, emotion_score, current_img, report_id, region, clipped_image):
    """
    this is used to generate the most dominant emotions among all other emotions available
    this is meant for male
    :return:
    """
    report = Report.objects.filter(id=report_id).first()
    if not report:
        print("No report found")
        return False

    # Update the most angry image if applicable
    if emotion_label == 'angry':
        if emotion_score > report.male_most_angry_score or str(
                report.male_most_angry_image) == "":
            report.male_most_angry_image = current_img
            report.male_most_angry_score = emotion_score
            report.male_most_angry_time = timezone.now()
            # save the regions also
            report.male_most_angry_face_x = region["x"]
            report.male_most_angry_face_y = region["y"]
            report.male_most_angry_face_width = region["w"]
            report.male_most_angry_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["male_most_angry_image"],
                                 clipped_image=clipped_image)

    # Update the report.male_most disgust image if applicable
    if emotion_label == 'disgust':
        if emotion_score > report.male_most_disgust_score or str(
                report.male_most_disgust_image) == "":
            report.male_most_disgust_image = current_img
            report.male_most_disgust_score = emotion_score
            report.male_most_disgust_time = timezone.now()
            # save the regions also
            report.male_most_disgust_face_x = region["x"]
            report.male_most_disgust_face_y = region["y"]
            report.male_most_disgust_face_width = region["w"]
            report.male_most_disgust_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["male_most_disgust_image"],
                                 clipped_image=clipped_image)

    # Update the report.male_most happy image if applicable
    if emotion_label == 'happy':
        if emotion_score > report.male_most_happy_score or str(
                report.male_most_happy_image) == "":
            report.male_most_happy_image = current_img
            report.male_most_happy_score = emotion_score
            report.male_most_happy_time = timezone.now()
            # save the regions also
            report.male_most_happy_face_x = region["x"]
            report.male_most_happy_face_y = region["y"]
            report.male_most_happy_face_width = region["w"]
            report.male_most_happy_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["male_most_happy_image"],
                                 clipped_image=clipped_image)

    # Update the report.male_most fear image if applicable
    if emotion_label == 'fear':
        if emotion_score > report.male_most_fear_score or str(
                report.male_most_fear_image) == "":
            report.male_most_fear_image = current_img
            report.male_most_fear_score = emotion_score
            report.male_most_fear_time = timezone.now()
            # save the regions also
            report.male_most_fear_face_x = region["x"]
            report.male_most_fear_face_y = region["y"]
            report.male_most_fear_face_width = region["w"]
            report.male_most_fear_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["male_most_fear_image"],
                                 clipped_image=clipped_image)

    # Update the report.male_most sad image if applicable
    if emotion_label == 'sad':
        if emotion_score > report.male_most_sad_score or str(report.male_most_sad_image) == "":
            report.male_most_sad_image = current_img
            report.male_most_sad_score = emotion_score
            report.male_most_sad_time = timezone.now()
            # save the regions also
            report.male_most_sad_face_x = region["x"]
            report.male_most_sad_face_y = region["y"]
            report.male_most_sad_face_width = region["w"]
            report.male_most_sad_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["male_most_sad_image"],
                                 clipped_image=clipped_image)

    # Update the report.male_most surprise image if applicable
    if emotion_label == 'surprise':
        if emotion_score > report.male_most_surprise_score or str(
                report.male_most_surprise_image) == "":
            report.male_most_surprise_image = current_img
            report.male_most_surprise_score = emotion_score
            report.male_most_surprise_time = timezone.now()
            # save the regions also
            report.male_most_surprise_face_x = region["x"]
            report.male_most_surprise_face_y = region["y"]
            report.male_most_surprise_face_width = region["w"]
            report.male_most_surprise_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["male_most_surprise_image"],
                                 clipped_image=clipped_image)

    # Update the report.male_most neutral image if applicable
    if emotion_label == 'neutral':
        if emotion_score > report.male_most_neutral_score or str(
                report.male_most_neutral_image) == "":
            report.male_most_neutral_image = current_img
            report.male_most_neutral_score = emotion_score
            report.male_most_neutral_time = timezone.now()
            # save the regions also
            report.male_most_neutral_face_x = region["x"]
            report.male_most_neutral_face_y = region["y"]
            report.male_most_neutral_face_width = region["w"]
            report.male_most_neutral_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["male_most_neutral_image"],
                                 clipped_image=clipped_image)

    report.save()
    return True


@shared_task
def generate_female_most_dominant_emotion(emotion_label, emotion_score, current_img, report_id, region, clipped_image):
    """
    this is used to generate the most dominant emotions among all other emotions available
    this is meant for male
    :return:
    """
    report = Report.objects.filter(id=report_id).first()

    if not report:
        print("No report found")
        return False

    # Update the most angry image if applicable
    if emotion_label == 'angry':
        if emotion_score > report.female_most_angry_score or str(
                report.female_most_angry_image) == "":
            report.female_most_angry_image = current_img
            report.female_most_angry_score = emotion_score
            report.female_most_angry_time = timezone.now()
            # save the regions also
            report.female_most_angry_face_x = region["x"]
            report.female_most_angry_face_y = region["y"]
            report.female_most_angry_face_width = region["w"]
            report.female_most_angry_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["male_most_angry_image"],
                                 clipped_image=clipped_image)

    # Update the report.female_most disgust image if applicable
    if emotion_label == 'disgust':
        if emotion_score > report.female_most_disgust_score or str(
                report.female_most_disgust_image) == "":
            report.female_most_disgust_image = current_img
            report.female_most_disgust_score = emotion_score
            report.female_most_disgust_time = timezone.now()
            # save the regions also
            report.female_most_disgust_face_x = region["x"]
            report.female_most_disgust_face_y = region["y"]
            report.female_most_disgust_face_width = region["w"]
            report.female_most_disgust_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["male_most_angry_image"],
                                 clipped_image=clipped_image)

    # Update the report.female_most happy image if applicable
    if emotion_label == 'happy':
        if emotion_score > report.female_most_happy_score or str(
                report.female_most_happy_image) == "":
            report.female_most_happy_image = current_img
            report.female_most_happy_score = emotion_score
            report.female_most_happy_time = timezone.now()
            # save the regions also
            report.female_most_happy_face_x = region["x"]
            report.female_most_happy_face_y = region["y"]
            report.female_most_happy_face_width = region["w"]
            report.female_most_happy_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["female_most_happy_image"],
                                 clipped_image=clipped_image)

    # Update the report.female_most fear image if applicable
    if emotion_label == 'fear':
        if emotion_score > report.female_most_fear_score or str(
                report.female_most_fear_image) == "":
            report.female_most_fear_image = current_img
            report.female_most_fear_score = emotion_score
            report.female_most_fear_time = timezone.now()
            # save the regions also
            report.female_most_fear_face_x = region["x"]
            report.female_most_fear_face_y = region["y"]
            report.female_most_fear_face_width = region["w"]
            report.female_most_fear_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["female_most_fear_image"],
                                 clipped_image=clipped_image)

    # Update the report.female_most sad image if applicable
    if emotion_label == 'sad':
        if emotion_score > report.female_most_sad_score or str(
                report.female_most_sad_image) == "":
            report.female_most_sad_image = current_img
            report.female_most_sad_score = emotion_score
            report.female_most_sad_time = timezone.now()
            # save the regions also
            report.female_most_sad_face_x = region["x"]
            report.female_most_sad_face_y = region["y"]
            report.female_most_sad_face_width = region["w"]
            report.female_most_sad_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["female_most_sad_image"],
                                 clipped_image=clipped_image)

    # Update the report.female_most surprise image if applicable
    if emotion_label == 'surprise':
        if emotion_score > report.female_most_surprise_score or str(
                report.female_most_surprise_image) == "":
            report.female_most_surprise_image = current_img
            report.female_most_surprise_score = emotion_score
            report.female_most_surprise_time = timezone.now()
            # save the regions also
            report.female_most_surprise_face_x = region["x"]
            report.female_most_surprise_face_y = region["y"]
            report.female_most_surprise_face_width = region["w"]
            report.female_most_surprise_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["female_most_surprise_image"],
                                 clipped_image=clipped_image)

    # Update the report.female_most neutral image if applicable
    if emotion_label == 'neutral':
        if emotion_score > report.female_most_neutral_score or str(
                report.female_most_neutral_image) == "":
            report.female_most_neutral_image = current_img
            report.female_most_neutral_score = emotion_score
            report.female_most_neutral_time = timezone.now()
            # save the regions also
            report.female_most_neutral_face_x = region["x"]
            report.female_most_neutral_face_y = region["y"]
            report.female_most_neutral_face_width = region["w"]
            report.female_most_neutral_face_height = region["h"]

            report.save()
            # check if the image exists
            verify_image_if_same(report=report, exclude_name=["female_most_neutral_image"],
                                 clipped_image=clipped_image)

    report.save()
    return True


@shared_task
def generate_data(image, report_id):
    # clear the session first
    tf.compat.v1.keras.backend.clear_session()

    image = np.array(image, dtype=np.uint8)

    report = Report.objects.filter(id=report_id).first()
    if not report:
        print("No report found")
        return False

    # Use DeepFace to detect emotions in the image
    try:
        print("analyzing")
        # clear the session first
        tf.compat.v1.keras.backend.clear_session()
        detected_faces = DeepFace.analyze(image, actions=('emotion', 'gender',), enforce_detection=True)
        # clear the session last
        tf.compat.v1.keras.backend.clear_session()

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
                # Create a copy of the image to draw the rectangle and add text because we are gonna make a loop
                current_img = image.copy()

                if 'region' in face:
                    # Get the region of the face coordinate
                    emotions = face['emotion']
                    emotion_label = max(emotions, key=emotions.get)
                    emotion_score = emotions[emotion_label]
                    emotion_score = round(emotion_score, 2)
                    # make a new image
                    face_region = image[y:y + h, x:x + w]

                    # Resize the extracted face region to 400 by 400 pixels which is gonna be used to verify if the image exist before on other
                    #  emotions
                    clipped_image = cv2.resize(face_region, (400, 400))
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
                        report.angry_count += emotions["angry"]
                    elif emotion_label == "disgust":
                        report.disgust_count += emotions["disgust"]
                    elif emotion_label == "fear":
                        report.fear_count += emotions["fear"]
                    elif emotion_label == "happy":
                        report.happy_count += emotions["happy"]
                    elif emotion_label == "sad":
                        report.sad_count += emotions["sad"]
                    elif emotion_label == "surprise":
                        report.surprise_count += emotions["surprise"]
                    elif emotion_label == "neutral":
                        report.neutral_count += emotions["neutral"]
                    report.save()

                    dominant_gender = face.get("dominant_gender")

                    #  Only generate data if the emotion score is greater than 70
                    print("The emotion score: ", emotion_score)
                    if emotion_score > 50:
                        # convert the image to temporary image for django to use
                        current_img = convert_opencv_to_image(current_img)

                        print("Dominant reached here")
                        # update the dominant emotion
                        generate_most_dominant_emotion(emotion_label, emotion_score, current_img, report_id,
                                                       face['region'], clipped_image)
                        if dominant_gender == "Man":
                            generate_male_most_dominant_emotion(emotion_label, emotion_score, current_img,
                                                                report_id, face['region'], clipped_image)
                        elif dominant_gender == "Woman":
                            generate_female_most_dominant_emotion(emotion_label, emotion_score,
                                                                  current_img, report_id, face['region'], clipped_image)

    except Exception as a:
        # clear the session
        tf.compat.v1.keras.backend.clear_session()
        print(a)
    return
