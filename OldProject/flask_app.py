import os
import sqlite3

from flask import Flask, render_template
from decouple import config

ProjectName = config("PROJECT_NAME")

app = Flask(__name__, template_folder="report", static_folder=ProjectName)
# Set the path to the image directory
IMAGE_DIR = os.path.join(app.root_path, 'static', ProjectName)


@app.route('/')
def render_report():
    # Query the data from the database
    conn = sqlite3.connect(f'{ProjectName}/db.sqlite')
    cur = conn.cursor()
    try:
        dominant_emotions_data = cur.execute(
            "SELECT emotion_label,dominant_frame,score FROM dominant_emotions GROUP BY emotion_label"
        ).fetchall()
    except:
        dominant_emotions_data = []
    try:
        male_dominant_emotions_data = cur.execute(
            "SELECT emotion_label,dominant_frame,score FROM male_dominant_emotions GROUP BY emotion_label"
        ).fetchall()
    except:
        male_dominant_emotions_data = []

    try:
        female_dominant_emotions_data = cur.execute(
            "SELECT emotion_label,dominant_frame,score FROM female_dominant_emotions GROUP BY emotion_label"
        ).fetchall()
    except:
        female_dominant_emotions_data = []
    pie_chart_image = f"{ProjectName}/chart.png"
    path = os.getcwd()

    # Render the HTML template with the data
    return render_template(
        'template.html',
        dominant_emotions_data=dominant_emotions_data,
        male_dominant_emotions_data=male_dominant_emotions_data,
        female_dominant_emotions_data=female_dominant_emotions_data,
        pie_chart_image=pie_chart_image,
        path=path,
        image_dir=IMAGE_DIR
    )


if __name__ == '__main__':
    app.run()
