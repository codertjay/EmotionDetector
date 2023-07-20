import asyncio

from tasks import *
import cv2
import time


async def process_frame(frame):
    print("create task now")
    generate_data.delay(frame.tolist())
    return True


def run_emotion_detection():
    """
    This runs the emotion detection
    :return:
    """
    # Load the DeepFace model
    model = DeepFace.build_model('Emotion', )
    # Set up the timer
    timer_duration = 10  # seconds
    last_face_sent_time = time.time()

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
            current_time = time.time()
            elapsed_time = current_time - last_face_sent_time
            if elapsed_time >= timer_duration:
                print("Creating task")
                asyncio.create_task(process_frame(frame))
                last_face_sent_time = current_time

        cv2.imshow('frame', resized)

        # Quit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close
    cap.release()
    cv2.destroyAllWindows()


loop = asyncio.get_event_loop()
loop.run_until_complete(run_emotion_detection())
