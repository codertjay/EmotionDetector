from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views import View
from django.views.generic import ListView

from emotions.models import Report


# Create your views here.

class DashboardView(View):

    def get(self, request):
        reports = Report.objects.filter(user=self.request.user)

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
            "disgust_count": disgust_count,
            "angry_count": angry_count,
            "happy_count": happy_count,
            "fear_count": fear_count,
            "sad_count": sad_count,
            "surprise_count": surprise_count,
            "neutral_count": neutral_count,
        }
        return render(request, "dashboard.html", context)
