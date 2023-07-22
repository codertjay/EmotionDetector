from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render
from django.views import View

from emotions.models import Report


# Create your views here.

class DashboardView(LoginRequiredMixin, View):

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
            "disgust_count": round(disgust_count, 2),
            "angry_count": round(angry_count, 2),
            "happy_count": round(happy_count, 2),
            "fear_count": round(fear_count, 2),
            "sad_count": round(sad_count, 2),
            "surprise_count": round(surprise_count, 2),
            "neutral_count": round(neutral_count, 2),
        }
        return render(request, "dashboard.html", context)
