import base64
import json

import cv2
import numpy as np
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.views import View
from django.views.generic import ListView, DetailView

from emotions.models import Report
from emotions.tasks import generate_data
from emotions.utils import resize_cv2_image


class ReportCreateView(LoginRequiredMixin, View):

    def get(self, request):
        return render(request, "report_create.html")

    def post(self, request):
        name = self.request.POST.get("name")
        if not name:
            messages.error(request, "Error creating class")
            return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
        report = Report.objects.create(name=name, user=self.request.user)
        messages.success(request, "Successfully start class")
        return redirect("emotion_detection", report.id)


class ReportListView(LoginRequiredMixin, ListView):
    paginate_by = 10
    queryset = Report.objects.all()
    template_name = "index.html"
    context_object_name = "reports"

    def get_queryset(self):
        report = Report.objects.filter(user=self.request.user)
        return report


class ReportDetailView(LoginRequiredMixin, DetailView):
    paginate_by = 10
    queryset = Report.objects.all()
    template_name = "report_detail.html"
    context_object_name = "report"

    def get_queryset(self):
        report = Report.objects.filter(user=self.request.user)
        return report


class ReportDeleteView(LoginRequiredMixin, View):
    """
    this is used to delete a Report
    """

    def post(self, request):
        #  this  deletes a redirect back to the page
        item_id = request.POST.get("report_id")
        if item_id:
            report = Report.objects.filter(id=item_id, user=self.request.user).first()
            if report:
                report.delete()
        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


def emotion_detection(request, id):
    context = {
        "id": id
    }
    return render(request, 'emotion_detection.html', context)


def report_stat_view(request, id):
    """
     this is used to get the current status of the report which is currently life
     in which a request  be comming from the front end to the backend every  minutes
     """

    report = Report.objects.filter(id=id).first()
    if not report:
        return JsonResponse({"error": "No report with this id"}, status=400)
    data = {
        "disgust": report.percentage_disgust,
        "angry": report.percentage_angry,
        "happy": report.percentage_happy,
        "fear": report.percentage_fear,
        "sad": report.percentage_sad,
        "surprise": report.percentage_surprise,
        "neutral": report.percentage_neutral,
    }
    return JsonResponse(data, status=200)


def report_automated_view(request, id):
    """
    This view is use to show  more info compare to the stat like the pie chart
    """
    report = Report.objects.filter(id=id).first()
    if not report:
        return JsonResponse({"error": "No report with this id"}, status=400)
    pie_chart = f"{request.get_host()}{report.chartImageURL()}"
    data = {
        "disgust": report.percentage_disgust,
        "angry": report.percentage_angry,
        "happy": report.percentage_happy,
        "fear": report.percentage_fear,
        "sad": report.percentage_sad,
        "surprise": report.percentage_surprise,
        "neutral": report.percentage_neutral,
        "pie_chart": pie_chart,
    }
    return JsonResponse(data, status=200)


def process_video_frame(request, id):
    #  get the report from the id passed
    report = Report.objects.filter(id=id).first()
    if not report:
        return JsonResponse({"error": "Report Does not exists"}, status=400)

    if request.method == 'POST':
        body = json.loads(request.body)
        frame_data = body.get("frame")
        if frame_data:
            # Convert the frame data from Base64 to OpenCV image format
            _, img_encoded = frame_data.split(';base64,')
            np_array = np.frombuffer(base64.b64decode(img_encoded), dtype=np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            # resize the image
            # img = cv2.resize(img, (1000, 1000))
            img = resize_cv2_image(img, 750, 750)

            # for debugging
            # cv2.imwrite("media.jpeg", img)
            # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # plt.imshow(img_rgb)
            # plt.show()
            # Perform additional processing on the image as needed
            # ...
            generate_data.delay(img.tolist(), report.id)
            # generate_data.delay(img.tolist(), report.id, report.user.id)

            # Example: Convert the processed image back to Base64 for sending back to the browser
            # _, buffer = cv2.imencode('.jpg', img)
            # frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Return the processed frame as JSON response
            return JsonResponse({'processed_frame': "frame_base64"})

        return JsonResponse({'error': 'Invalid request'})
