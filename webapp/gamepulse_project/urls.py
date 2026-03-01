from django.urls import path, include

urlpatterns = [
    path("", include("core.urls")),
    path("copilot/", include("copilot.urls")),
]
