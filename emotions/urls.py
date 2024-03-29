from django.urls import path
from .views import emotion_detection, process_video_frame, ReportListView, ReportDeleteView, ReportCreateView, \
    ReportDetailView, report_stat_view, report_automated_view

urlpatterns = [
    path('emotion-detection/<int:id>/', emotion_detection, name='emotion_detection'),
    path('process-video-frame/<int:id>/', process_video_frame, name='process_video_frame'),
    path('report_create', ReportCreateView.as_view(), name='report_create'),
    path('report_list', ReportListView.as_view(), name='report_list'),
    path('report_detail/<int:pk>/', ReportDetailView.as_view(), name='report_detail'),
    path('report_delete', ReportDeleteView.as_view(), name='report_delete'),
    path('report_stat/<int:id>/', report_stat_view, name='report_stat'),
    path('report_automated/<int:id>/', report_automated_view, name='report_automated'),
]
