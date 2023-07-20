import asyncio
import asyncio

from tasks import *
import cv2
import time

loop = asyncio.get_event_loop()


async def process_frame(frame):
    print("create task now")
    generate_data.delay(frame.tolist())


async def run_emotion_detection():
    """
    This runs the emotion detection
    :return:
    """
    # Load the DeepFace model
    model = DeepFace.build_model('Emotion', )

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Start the video stream
    if config("USE_MOBILE", cast=bool):
        cap = cv2.VideoCapture(config("MOBILE_IP"))
    else:
        cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # Read a frame from the video stream
        ret, frame = cap.read()

        # Display the frame
        resized = cv2.resize(frame, (600, 400))

        # Detect faces using Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Check if there are any faces detected in the frame
        if len(faces) > 0:
            print("Processing Frame")
            generate_data.delay(frame.tolist())

        cv2.imshow('frame', resized)

        # Quit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close
    cap.release()
    cv2.destroyAllWindows()


loop.run_until_complete(run_emotion_detection())
loop.close()
