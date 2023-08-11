# EmotionDetector

This is used to detect emeotio and generate a report
Sure, here's a sample README for your Django project that you can use as a starting point. Make sure to replace
placeholders with actual information specific to your project:

---

# Emotion Detector Django Project

![Emotion Detector Logo](path_to_logo.png)

Welcome to the Emotion Detector Django project! This project aims to analyze and detect emotions from web-rct images and implementing to open cv.

## Table of Contents

- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python (version 3.11)
- Django (version 4.2.3)
- Other project dependencies listed in `requirements.txt`

### Installation

1. Clone this repository:

```bash
git clone https://github.com/codertjay/EmotionDetector.git
```

2. Navigate to the project directory:

```bash
cd EmotionDetector
```

3. Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

4. Create and apply the database migrations:

```bash
python manage.py makemigrations
python manage.py migrate
```

5. Run the development server:

```bash
python manage.py runserver
```

open a new terminal and run python3 to download the model

```
python3
from deepface import DeepFace

# download the deepface model
DeepFace.build_model("Gender")
DeepFace.build_model("Emotion")
DeepFace.build_model("VGG-Face")

```

6. Open your web browser and navigate to `http://127.0.0.1:8000/` to access the Emotion Detector app.

## Usage

1. On the home page, enter a piece of text in the input field and click the "Analyze" button.
2. The app will process the text and display the detected emotion along with a confidence score.

Feel free to explore the app and its features further!

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Open a pull request describing your changes.

---

| Task Name                                                  | Week 1   | Week 2   | Week 3   | Week 4   | Week 5   | Week 6   | Week 7   | Week 8   | Week 9   |
|------------------------------------------------------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| 1. Project planning and scoping                            | ✅        | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 2. Research on road sign detection and recognition systems | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 3. Data Collection and Preprocessing                       | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 4. Trainging and testing of deep learning models           | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 5. Integration with Web application and hardware testing   | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 6. User Interface development                              | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 7. System integration and testing                          | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 8. Documentation and negotiation                           | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |

Remember to customize the README according to your project's specific details, including installation steps, usage
instructions, and any additional information you want to provide to users and potential contributors.
