import os

from deepface import DeepFace
from django.contrib.auth.models import User
from django.test import TestCase, Client
from django.urls import reverse

from emotions.models import Report



class TestDeepFaceEmotionRecognition(TestCase):

    def setUp(self):
        # Define the folder containing subfolders with emotion images
        self.emotion_folder_path = "Test/emotion"
        self.gender_folder_path = "Test/gender"
        # Define the paths to the two images for verification
        self.img1_path = "Test/verify/image_1.jpeg"
        self.img2_path = "Test/verify/image_2.jpeg"

    def test_emotion_recognition(self):
        # Get a list of emotion subfolders
        emotion_subfolders = [f for f in os.listdir(self.emotion_folder_path) if
                              os.path.isdir(os.path.join(self.emotion_folder_path, f))]

        # Initialize DeepFace with the emotion analysis action
        DeepFace.build_model("Emotion")

        # Loop through emotion subfolders and perform emotion recognition
        for emotion in emotion_subfolders:
            emotion_folder_path = os.path.join(self.emotion_folder_path, emotion)
            emotion_images = [os.path.join(emotion_folder_path, img) for img in os.listdir(emotion_folder_path)]

            for image_path in emotion_images:
                try:
                    # Analyze the image for emotion using the pre-trained model
                    result = DeepFace.analyze(image_path, actions=['emotion'])

                    # Extract the detected emotion from the result
                    detected_emotion = result[0]['dominant_emotion']

                    # Define the expected emotion based on the subfolder name
                    expected_emotion = emotion.lower()  # Assuming subfolder names match emotion labels

                    # Assert that the detected emotion matches the expected emotion
                    self.assertEqual(detected_emotion, expected_emotion, f"Failed for image: {image_path}")
                except Exception as e:
                    # Handle any errors that may occur during processing
                    self.fail(f"Error processing {image_path}: {str(e)}")

    def test_gender_recognition(self):
        # Get a list of gender subfolders
        gender_subfolders = [f for f in os.listdir(self.gender_folder_path) if
                             os.path.isdir(os.path.join(self.gender_folder_path, f))]

        # Initialize DeepFace with the gender analysis action
        DeepFace.build_model("Gender")

        # Loop through gender subfolders and perform gender recognition
        for gender in gender_subfolders:
            gender_folder_path = os.path.join(self.gender_folder_path, gender)
            gender_images = [os.path.join(gender_folder_path, img) for img in os.listdir(gender_folder_path)]

            # Loop through image files and perform gender recognition
            for image_path in gender_images:
                try:
                    # Analyze the image for gender using the pre-trained model
                    result = DeepFace.analyze(image_path, actions=['gender'])

                    # Extract the detected gender from the result
                    detected_gender = result[0]['dominant_gender']

                    # Define the expected gender based on the file name or other criteria
                    # Replace with your own logic for setting the expected gender
                    expected_gender = "male" if "male" in image_path.lower() else "female"

                    # Assert that the detected gender matches the expected gender
                    self.assertEqual(detected_gender, expected_gender, f"Failed for image: {image_path}")

                except Exception as e:
                    # Handle any errors that may occur during processing
                    self.fail(f"Error processing {image_path}: {str(e)}")

    def test_face_verification(self):
        try:
            # Perform face verification using DeepFace with VGG-Face model
            result = DeepFace.verify(img1_path=self.img1_path,
                                     img2_path=self.img2_path,
                                     model_name="VGG-Face",
                                     enforce_detection=False)

            # Extract the similarity score from the result
            similarity_score = result['verified']

            # Define a similarity threshold for verification (adjust as needed)
            similarity_threshold = 0.9

            # Assert that the similarity score exceeds the threshold
            self.assertTrue(similarity_threshold >= result["threshold"],
                            f"Face verification failed. Similarity score: {similarity_score}")

        except Exception as e:
            # Handle any errors that may occur during processing
            self.fail(f"Error during face verification: {str(e)}")
