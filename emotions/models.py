import os

from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import pre_save
from matplotlib import pyplot as plt

from emotions.utils import create_slug


# Create your models here.


class Report(models.Model):
    """
    This report is generated on every project created for a class which is needed to be streamed
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=250, blank=True, null=True)
    slug = models.SlugField(max_length=500, blank=True, null=True)

    status = models.CharField(max_length=250, default="PENDING", choices=(
        ("DONE", "DONE"),
        ("PENDING", "PENDING"),
    ))
    #  the count for generating the pie chart
    disgust_count = models.IntegerField(default=0)
    angry_count = models.IntegerField(default=0)
    happy_count = models.IntegerField(default=0)
    fear_count = models.IntegerField(default=0)
    sad_count = models.IntegerField(default=0)
    surprise_count = models.IntegerField(default=0)
    neutral_count = models.IntegerField(default=0)
    chart_image = models.ImageField(blank=True, null=True)

    most_angry_image = models.ImageField(upload_to="images", blank=True, null=True)
    most_angry_score = models.FloatField(blank=True, null=True, default=0.0)
    most_angry_time = models.DateTimeField(blank=True, null=True)
    most_angry_face_x = models.IntegerField(blank=True, null=True)
    most_angry_face_y = models.IntegerField(blank=True, null=True)
    most_angry_face_width = models.IntegerField(blank=True, null=True)
    most_angry_face_height = models.IntegerField(blank=True, null=True)

    most_disgust_image = models.ImageField(upload_to="images", blank=True, null=True)
    most_disgust_score = models.FloatField(blank=True, null=True, default=0.0)
    most_disgust_time = models.DateTimeField(blank=True, null=True)
    most_disgust_face_x = models.IntegerField(blank=True, null=True)
    most_disgust_face_y = models.IntegerField(blank=True, null=True)
    most_disgust_face_width = models.IntegerField(blank=True, null=True)
    most_disgust_face_height = models.IntegerField(blank=True, null=True)

    most_happy_image = models.ImageField(upload_to="images", blank=True, null=True)
    most_happy_score = models.FloatField(blank=True, null=True, default=0.0)
    most_happy_time = models.DateTimeField(blank=True, null=True)
    most_happy_face_x = models.IntegerField(blank=True, null=True)
    most_happy_face_y = models.IntegerField(blank=True, null=True)
    most_happy_face_width = models.IntegerField(blank=True, null=True)
    most_happy_face_height = models.IntegerField(blank=True, null=True)

    most_fear_image = models.ImageField(upload_to="images", blank=True, null=True)
    most_fear_score = models.FloatField(blank=True, null=True, default=0.0)
    most_fear_time = models.DateTimeField(blank=True, null=True)
    most_fear_face_x = models.IntegerField(blank=True, null=True)
    most_fear_face_y = models.IntegerField(blank=True, null=True)
    most_fear_face_width = models.IntegerField(blank=True, null=True)
    most_fear_face_height = models.IntegerField(blank=True, null=True)

    most_sad_image = models.ImageField(upload_to="images", blank=True, null=True)
    most_sad_score = models.FloatField(blank=True, null=True, default=0.0)
    most_sad_time = models.DateTimeField(blank=True, null=True)
    most_sad_face_x = models.IntegerField(blank=True, null=True)
    most_sad_face_y = models.IntegerField(blank=True, null=True)
    most_sad_face_width = models.IntegerField(blank=True, null=True)
    most_sad_face_height = models.IntegerField(blank=True, null=True)

    most_surprise_image = models.ImageField(upload_to="images", blank=True, null=True)
    most_surprise_score = models.FloatField(blank=True, null=True, default=0.0)
    most_surprise_time = models.DateTimeField(blank=True, null=True)
    most_surprise_face_x = models.IntegerField(blank=True, null=True)
    most_surprise_face_y = models.IntegerField(blank=True, null=True)
    most_surprise_face_width = models.IntegerField(blank=True, null=True)
    most_surprise_face_height = models.IntegerField(blank=True, null=True)

    most_neutral_image = models.ImageField(upload_to="images", blank=True, null=True)
    most_neutral_score = models.FloatField(blank=True, null=True, default=0.0)
    most_neutral_time = models.DateTimeField(blank=True, null=True)
    most_neutral_face_x = models.IntegerField(blank=True, null=True)
    most_neutral_face_y = models.IntegerField(blank=True, null=True)
    most_neutral_face_width = models.IntegerField(blank=True, null=True)
    most_neutral_face_height = models.IntegerField(blank=True, null=True)

    male_most_angry_image = models.ImageField(upload_to="images", blank=True, null=True)
    male_most_angry_score = models.FloatField(blank=True, null=True, default=0.0)
    male_most_angry_time = models.DateTimeField(blank=True, null=True)
    male_most_angry_face_x = models.IntegerField(blank=True, null=True)
    male_most_angry_face_y = models.IntegerField(blank=True, null=True)
    male_most_angry_face_width = models.IntegerField(blank=True, null=True)
    male_most_angry_face_height = models.IntegerField(blank=True, null=True)

    male_most_disgust_image = models.ImageField(upload_to="images", blank=True, null=True)
    male_most_disgust_score = models.FloatField(blank=True, null=True, default=0.0)
    male_most_disgust_time = models.DateTimeField(blank=True, null=True)
    male_most_disgust_face_x = models.IntegerField(blank=True, null=True)
    male_most_disgust_face_y = models.IntegerField(blank=True, null=True)
    male_most_disgust_face_width = models.IntegerField(blank=True, null=True)
    male_most_disgust_face_height = models.IntegerField(blank=True, null=True)

    male_most_happy_image = models.ImageField(upload_to="images", blank=True, null=True)
    male_most_happy_score = models.FloatField(blank=True, null=True, default=0.0)
    male_most_happy_time = models.DateTimeField(blank=True, null=True)
    male_most_happy_face_x = models.IntegerField(blank=True, null=True)
    male_most_happy_face_y = models.IntegerField(blank=True, null=True)
    male_most_happy_face_width = models.IntegerField(blank=True, null=True)
    male_most_happy_face_height = models.IntegerField(blank=True, null=True)

    male_most_fear_image = models.ImageField(upload_to="images", blank=True, null=True)
    male_most_fear_score = models.FloatField(blank=True, null=True, default=0.0)
    male_most_fear_time = models.DateTimeField(blank=True, null=True)
    male_most_fear_face_x = models.IntegerField(blank=True, null=True)
    male_most_fear_face_y = models.IntegerField(blank=True, null=True)
    male_most_fear_face_width = models.IntegerField(blank=True, null=True)
    male_most_fear_face_height = models.IntegerField(blank=True, null=True)

    male_most_sad_image = models.ImageField(upload_to="images", blank=True, null=True)
    male_most_sad_score = models.FloatField(blank=True, null=True, default=0.0)
    male_most_sad_time = models.DateTimeField(blank=True, null=True)
    male_most_sad_face_x = models.IntegerField(blank=True, null=True)
    male_most_sad_face_y = models.IntegerField(blank=True, null=True)
    male_most_sad_face_width = models.IntegerField(blank=True, null=True)
    male_most_sad_face_height = models.IntegerField(blank=True, null=True)

    male_most_surprise_image = models.ImageField(upload_to="images", blank=True, null=True)
    male_most_surprise_score = models.FloatField(blank=True, null=True, default=0.0)
    male_most_surprise_time = models.DateTimeField(blank=True, null=True)
    male_most_surprise_face_x = models.IntegerField(blank=True, null=True)
    male_most_surprise_face_y = models.IntegerField(blank=True, null=True)
    male_most_surprise_face_width = models.IntegerField(blank=True, null=True)
    male_most_surprise_face_height = models.IntegerField(blank=True, null=True)

    male_most_neutral_image = models.ImageField(upload_to="images", blank=True, null=True)
    male_most_neutral_score = models.FloatField(blank=True, null=True, default=0.0)
    male_most_neutral_time = models.DateTimeField(blank=True, null=True)
    male_most_neutral_face_x = models.IntegerField(blank=True, null=True)
    male_most_neutral_face_y = models.IntegerField(blank=True, null=True)
    male_most_neutral_face_width = models.IntegerField(blank=True, null=True)
    male_most_neutral_face_height = models.IntegerField(blank=True, null=True)

    female_most_angry_image = models.ImageField(upload_to="images", blank=True, null=True)
    female_most_angry_score = models.FloatField(blank=True, null=True, default=0.0)
    female_most_angry_time = models.DateTimeField(blank=True, null=True)
    female_most_angry_face_x = models.IntegerField(blank=True, null=True)
    female_most_angry_face_y = models.IntegerField(blank=True, null=True)
    female_most_angry_face_width = models.IntegerField(blank=True, null=True)
    female_most_angry_face_height = models.IntegerField(blank=True, null=True)

    female_most_disgust_image = models.ImageField(upload_to="images", blank=True, null=True)
    female_most_disgust_score = models.FloatField(blank=True, null=True, default=0.0)
    female_most_disgust_time = models.DateTimeField(blank=True, null=True)
    female_most_disgust_face_x = models.IntegerField(blank=True, null=True)
    female_most_disgust_face_y = models.IntegerField(blank=True, null=True)
    female_most_disgust_face_width = models.IntegerField(blank=True, null=True)
    female_most_disgust_face_height = models.IntegerField(blank=True, null=True)

    female_most_happy_image = models.ImageField(upload_to="images", blank=True, null=True)
    female_most_happy_score = models.FloatField(blank=True, null=True, default=0.0)
    female_most_happy_time = models.DateTimeField(blank=True, null=True)
    female_most_happy_face_x = models.IntegerField(blank=True, null=True)
    female_most_happy_face_y = models.IntegerField(blank=True, null=True)
    female_most_happy_face_width = models.IntegerField(blank=True, null=True)
    female_most_happy_face_height = models.IntegerField(blank=True, null=True)

    female_most_fear_image = models.ImageField(upload_to="images", blank=True, null=True)
    female_most_fear_score = models.FloatField(blank=True, null=True, default=0.0)
    female_most_fear_time = models.DateTimeField(blank=True, null=True)
    female_most_fear_face_x = models.IntegerField(blank=True, null=True)
    female_most_fear_face_y = models.IntegerField(blank=True, null=True)
    female_most_fear_face_width = models.IntegerField(blank=True, null=True)
    female_most_fear_face_height = models.IntegerField(blank=True, null=True)

    female_most_sad_image = models.ImageField(upload_to="images", blank=True, null=True)
    female_most_sad_score = models.FloatField(blank=True, null=True, default=0.0)
    female_most_sad_time = models.DateTimeField(blank=True, null=True)
    female_most_sad_face_x = models.IntegerField(blank=True, null=True)
    female_most_sad_face_y = models.IntegerField(blank=True, null=True)
    female_most_sad_face_width = models.IntegerField(blank=True, null=True)
    female_most_sad_face_height = models.IntegerField(blank=True, null=True)

    female_most_surprise_image = models.ImageField(upload_to="images", blank=True, null=True)
    female_most_surprise_score = models.FloatField(blank=True, null=True, default=0.0)
    female_most_surprise_time = models.DateTimeField(blank=True, null=True)
    female_most_surprise_face_x = models.IntegerField(blank=True, null=True)
    female_most_surprise_face_y = models.IntegerField(blank=True, null=True)
    female_most_surprise_face_width = models.IntegerField(blank=True, null=True)
    female_most_surprise_face_height = models.IntegerField(blank=True, null=True)

    female_most_neutral_image = models.ImageField(upload_to="images", blank=True, null=True)
    female_most_neutral_score = models.FloatField(blank=True, null=True, default=0.0)
    female_most_neutral_time = models.DateTimeField(blank=True, null=True)
    female_most_neutral_face_x = models.IntegerField(blank=True, null=True)
    female_most_neutral_face_y = models.IntegerField(blank=True, null=True)
    female_most_neutral_face_width = models.IntegerField(blank=True, null=True)
    female_most_neutral_face_height = models.IntegerField(blank=True, null=True)

    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-id"]

    def save(self, *args, **kwargs):
        # Check if the object is being updated (has a primary key) and call the cleanup_images method
        if self.pk:
            # Fetch the original object from the database to compare changes
            original_report = Report.objects.get(pk=self.pk)

            # Check each image field for changes and clean up old image files
            image_fields = [
                'chart_image',
                'most_angry_image', 'most_disgust_image', 'most_happy_image', 'most_fear_image',
                'most_sad_image', 'most_surprise_image', 'most_neutral_image',
                'male_most_angry_image', 'male_most_disgust_image', 'male_most_happy_image', 'male_most_fear_image',
                'male_most_sad_image', 'male_most_surprise_image', 'male_most_neutral_image',
                'female_most_angry_image', 'female_most_disgust_image', 'female_most_happy_image',
                'female_most_fear_image',
                'female_most_sad_image', 'female_most_surprise_image', 'female_most_neutral_image',
            ]

            for field_name in image_fields:
                old_image_file = getattr(original_report, field_name, None)
                new_image_file = getattr(self, field_name, None)

                if old_image_file and new_image_file and old_image_file.path != new_image_file.path:
                    # Clean up the old image file if the field has changed
                    if os.path.isfile(old_image_file.path):
                        os.remove(old_image_file.path)
                    # Set the field to None if the new image file is removed
                    if not new_image_file:
                        setattr(self, field_name, None)

        # Call the original save method to save the changes to the database
        super(Report, self).save(*args, **kwargs)

    @property
    def total_count(self):
        return self.disgust_count + self.angry_count + self.happy_count + self.fear_count + self.sad_count + self.surprise_count + self.neutral_count

    @property
    def percentage_disgust(self):
        total = self.total_count
        if total > 0:
            return round((self.disgust_count / total) * 100, 2)
        return 0

    @property
    def percentage_angry(self):
        total = self.total_count
        if total > 0:
            return round((self.angry_count / total) * 100, 2)
        return 0

    @property
    def percentage_happy(self):
        total = self.total_count
        if total > 0:
            return round((self.happy_count / total) * 100, 2)
        return 0

    @property
    def percentage_fear(self):
        total = self.total_count
        if total > 0:
            return round((self.fear_count / total) * 100, 2)
        return 0

    @property
    def percentage_sad(self):
        total = self.total_count
        if total > 0:
            return round((self.sad_count / total) * 100, 2)
        return 0

    @property
    def percentage_surprise(self):
        total = self.total_count
        if total > 0:
            return round((self.surprise_count / total) * 100, 2)
        return 0

    @property
    def percentage_neutral(self):
        total = self.total_count
        if total > 0:
            return round((self.neutral_count / total) * 100, 2)
        return 0

    @property
    def generate_emotion_pie_chart(self):
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # Save the emotion counts in a pie chart
        emotion_counts = {
            "angry": self.angry_count,
            "disgust": self.disgust_count,
            "fear": self.fear_count,
            "happy": self.happy_count,
            "sad": self.sad_count,
            "surprise": self.surprise_count,
            "neutral": self.neutral_count
        }

        # Create a list of counts in the same order as the labels
        values = [emotion_counts[emotion] for emotion in emotions]

        # Remove the label with zero value from the lists
        non_zero_labels = []
        non_zero_values = []
        for label, value in zip(emotions, values):
            if value > 0:
                non_zero_labels.append(label)
                non_zero_values.append(value)

        labels = non_zero_labels
        values = non_zero_values

        # Set the figure size
        plt.figure(figsize=(8, 8))
        # Create the pie chart
        plt.pie(values, labels=labels, autopct='%1.1f%%', labeldistance=1.05, textprops={'fontsize': 14})
        plt.title('Emotions Pie Chart')

        # Save the image with a unique filename (you can use the timestamp or a unique identifier)
        chart_image_filename = f"{self.slug}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.png"
        if not os.path.exists("media/images"):
            os.mkdir("media/images")
        chart_image_path = os.path.join("media/images", chart_image_filename)
        plt.savefig(chart_image_path)

        # Save the image path to the chart_image field of the model instance
        self.chart_image = chart_image_path.replace("media/", "")
        self.save()

        return True

    def chartImageURL(self):
        try:
            generated = self.generate_emotion_pie_chart
            imageUrl = self.chart_image.url
        except:
            imageUrl = None
        return imageUrl


def pre_save_event_receiver(sender, instance, *args, **kwargs):
    # enable creating slug for a  event before it is being saved
    if not instance.slug:
        instance.slug = create_slug(instance, Report)


pre_save.connect(pre_save_event_receiver, sender=Report)
