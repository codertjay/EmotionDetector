from django.urls import path

from .views import DashboardView
from .views import UserProfileView, ChangeUserPassword,custom_logout

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path("user_profile/", UserProfileView.as_view(), name="user_profile"),
    path("change_user_password/", ChangeUserPassword.as_view(), name="change_user_password"),
    path("custom_logout", custom_logout, name="custom_logout"),

]
