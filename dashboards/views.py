from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.views import View

from dashboards.forms import UserProfileUpdateForm, ChangePasswordForm
from emotions.models import Report


# Create your views here.

class DashboardView(LoginRequiredMixin, View):

    def get(self, request):
        reports = Report.objects.filter(user=self.request.user)[:5]

        disgust_count = 0
        angry_count = 0
        happy_count = 0
        fear_count = 0
        sad_count = 0
        surprise_count = 0
        neutral_count = 0
        for item in reports:
            disgust_count += item.percentage_disgust
            angry_count += item.percentage_angry
            happy_count += item.percentage_happy
            fear_count += item.percentage_fear
            sad_count += item.percentage_sad
            surprise_count += item.percentage_surprise
            neutral_count += item.percentage_neutral

        context = {
            "reports": reports,
            "disgust_count": round(disgust_count, 2),
            "angry_count": round(angry_count, 2),
            "happy_count": round(happy_count, 2),
            "fear_count": round(fear_count, 2),
            "sad_count": round(sad_count, 2),
            "surprise_count": round(surprise_count, 2),
            "neutral_count": round(neutral_count, 2),
        }
        return render(request, "dashboard.html", context)


class UserProfileView(LoginRequiredMixin, View):
    """
    this is used to get the user profile and also update it
    """

    def get(self, request):
        user_profile = self.request.user.user_profile
        form = UserProfileUpdateForm(instance=user_profile)
        change_form = ChangePasswordForm()
        context = {
            "form": form,
            "change_form": change_form,
            "user_profile": user_profile,
        }
        return render(request, "user_profile.html", context)

    def post(self, request):
        user_profile = self.request.user.user_profile
        form = UserProfileUpdateForm(data=self.request.POST, files=self.request.FILES, instance=user_profile)
        if form.is_valid():
            form.save()
        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


class ChangeUserPassword(LoginRequiredMixin, View):

    def post(self, request):
        form = ChangePasswordForm(data=request.POST)
        if form.is_valid():
            password = form.cleaned_data.get("password")
            confirm_password = form.cleaned_data.get("confirm_password")
            if password != confirm_password:
                messages.error(request, "Password does not match")
                return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
            user = self.request.user
            user.set_password(password)
            user.save()
            user = authenticate(request, username=user.username, password=password,
                                backend='django.contrib.auth.backends.ModelBackend')
            if user is not None:
                login(request, user)
            messages.info(request, "Successfully Update password")
        else:
            for error in form.errors:
                messages.warning(request, f"{error}: {form.errors[error][0]}")
        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


def custom_logout(request):
    logout(request)
    return redirect('dashboard')  # Redirect to the desired URL after logout
