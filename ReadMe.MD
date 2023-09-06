# EmotionDetector

---

# Emotion Detector Django Project

![Emotion Detector Logo](staticfiles/img/logo-dark.png)

Welcome to the Emotion Detector Django project! This project aims to analyze and detect emotions from web-rct images and
implementing to open cv.

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

1. Extract the zip file:

```bash
Extract the zip file and open the terminal
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

open a new terminal and run python3 to download the model and make sure you are with in the virtual environment of the
project

```
python3
from deepface import DeepFace

# download the deepface model
DeepFace.build_model("Gender")
DeepFace.build_model("Emotion")
DeepFace.build_model("VGG-Face")

```

6. Redis Configuration

To install Redis on a Linux system, you can follow these general steps. The exact commands may vary slightly depending
on your Linux distribution. Here, I'll provide instructions for a few popular distributions: Ubuntu, CentOS, and Debian.

### Ubuntu

1. Update the package list to make sure you have the latest information about available packages:

   ```bash
   sudo apt update
   ```

2. Install Redis:

   ```bash
   sudo apt install redis-server
   ```

3. Start the Redis service and enable it to start on boot:

   ```bash
   sudo systemctl start redis-server
   sudo systemctl enable redis-server
   ```

4. Verify that Redis is running:

   ```bash
   sudo systemctl status redis-server
   ```

### CentOS

1. Update the package list:

   ```bash
   sudo yum update
   ```

2. Install Redis:

   ```bash
   sudo yum install redis
   ```

3. Start the Redis service and enable it to start on boot:

   ```bash
   sudo systemctl start redis
   sudo systemctl enable redis
   ```

4. Verify that Redis is running:

   ```bash
   sudo systemctl status redis
   ```

### Debian

1. Update the package list:

   ```bash
   sudo apt update
   ```

2. Install Redis:

   ```bash
   sudo apt install redis-server
   ```

3. Start the Redis service and enable it to start on boot:

   ```bash
   sudo systemctl start redis-server
   sudo systemctl enable redis-server
   ```

4. Verify that Redis is running:

   ```bash
   sudo systemctl status redis-server
   ```

After these steps, Redis should be installed and running on your Linux system. You can configure and use Redis as needed
for your specific use cases.

7. Run the celery worker: Open a new terminal and run the command below. Also make sure redis is active and running in
   the background

```bash
 celery -A EmotionDetector worker --loglevel=info
```

8. Open your web browser and navigate to `http://127.0.0.1:8000/` to access the Emotion Detector app.

---

| Task Name                                                  | Week 1   | Week 2   | Week 3   | Week 4   | Week 5   | Week 6   | Week 7   | Week 8   | Week 9   |
|------------------------------------------------------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| 1. Project planning and scoping                            | ✅        | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 2. Research on road sign detection and recognition systems | -------- | ✅        | ✅        | -------- | -------- | -------- | -------- | -------- | -------- |
| 3. Data Collection and Preprocessing                       | -------- | -------- | -------- | ✅        | ✅        | ✅        | -------- | -------- | -------- |
| 4. Build Project and testing of deep learning models       | -------- | -------- | -------- | -------- | -------- | -------- | ✅        | ✅        | ✅        |
| 5. Integration with Web application and hardware testing   | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 6. User Interface development                              | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 7. System integration and testing                          | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 8. Documentation and negotiation                           | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |

Remember to customize the README according to your project's specific details, including installation steps, usage
instructions, and any additional information you want to provide to users and potential contributors.
