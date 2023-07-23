from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save


# Create your models here.
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, blank=True, null=True, related_name="user_profile")
    profile_image = models.ImageField(upload_to="profile_image")
    timestamp = models.DateTimeField(auto_now_add=True)

    def profileImageURL(self):
        try:
            profile_image = self.profile_image.url
        except:
            profile_image = None
        return profile_image


def post_save_create_user_profile(sender, instance, *args, **kwargs):
    """
    This creates a user  profile once a user is being created
    :param instance:  the user created or updated
    """
    if instance:
        user_profile = UserProfile.objects.filter(user=instance).first()
        if not user_profile:
            UserProfile.objects.create(user=instance)


# Create your models here.


post_save.connect(post_save_create_user_profile, sender=User)
