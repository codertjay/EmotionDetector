import base64
import os
import random
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils.text import slugify
import time

from matplotlib import pyplot as plt


def create_slug(instance, instances, new_slug=None):
    """
    This creates a slug base on the instance provided and also if the slug exists it appends the id
    """
    slug = slugify(instance.name)
    if new_slug is not None:
        slug = new_slug
    qs = instances.objects.filter(slug=slug).order_by('-id')
    if qs.exists():
        new_slug = f'{slug}-{random.randint(10000, 999999)}'
        return create_slug(instance, instances, new_slug=new_slug)
    return slug


def base64_to_image(base64_string):
    # Remove the prefix from the base64 string, e.g., "data:image/jpeg;base64,"
    if base64_string.startswith('data:image/jpeg;base64,'):
        base64_string = base64_string.replace('data:image/jpeg;base64,', '')
    elif base64_string.startswith('data:image/png;base64,'):
        base64_string = base64_string.replace('data:image/png;base64,', '')

    # Decode the base64 string to bytes
    image_bytes = base64.b64decode(base64_string)

    # Open the image using PIL
    image = Image.open(BytesIO(image_bytes))

    # Save the image to the specified output file path
    image.save("output_file_path.jpg")
    return image


def resize_cv2_image(image, target_width, target_height):
    # Get the original width and height of the image
    height, width = image.shape[:2]

    # Calculate the aspect ratio of the original image
    aspect_ratio = width / float(height)

    # Calculate the new width and height while maintaining the aspect ratio
    if target_width / float(target_height) > aspect_ratio:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # Resize the image using the calculated new width and height
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


def convert_opencv_to_image(current_img):
    # Assuming you have the OpenCV image in a variable named 'current_img'

    # Generate a unique filename for the image
    added_path = str(time.time()).replace(".", "")
    image_filename = f"{added_path}.jpg"

    # Create the path to save the image file
    if not os.path.exists("media/images"):
        os.mkdir("media/images")
    image_path = os.path.join("media/images", image_filename)

    # Save the OpenCV image to the specified path
    cv2.imwrite(image_path, current_img)

    # Return the relative path to the saved image file
    return image_path.replace("media/", "")


# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return image
