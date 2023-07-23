from django import forms
from django.contrib.auth.models import User

from .models import UserProfile
from django.core.exceptions import ValidationError


class UserProfileUpdateForm(forms.ModelForm):
    first_name = forms.CharField(max_length=150)
    last_name = forms.CharField(max_length=150)
    username = forms.CharField(max_length=150)
    email = forms.EmailField()
    profile_image = forms.ImageField()

    class Meta:
        model = UserProfile
        fields = [
            "first_name",
            "last_name",
            "username",
            "email",
            "profile_image",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance.user:
            self.fields["first_name"].initial = self.instance.user.first_name
            self.fields["last_name"].initial = self.instance.user.last_name
            self.fields["username"].initial = self.instance.user.username
            self.fields["email"].initial = self.instance.user.email

    def clean_username(self):
        username = self.cleaned_data["username"]
        if self.instance.user.username != username:
            if User.objects.filter(username=username).exists():
                raise ValidationError("This username is already taken.")
        return username

    def clean_email(self):
        email = self.cleaned_data["email"]
        if self.instance.user.email != email:
            if User.objects.filter(email=email).exists():
                raise ValidationError("This email is already taken.")
        return email

    def save(self, commit=True):
        user = self.instance.user
        user.username = self.cleaned_data["username"]
        user.email = self.cleaned_data["email"]
        user.first_name = self.cleaned_data["first_name"]
        user.last_name = self.cleaned_data["last_name"]
        user.save()
        if commit:
            self.instance.save()
        return self.instance


class ChangePasswordForm(forms.Form):
    password = forms.SlugField()
    confirm_password = forms.SlugField()
