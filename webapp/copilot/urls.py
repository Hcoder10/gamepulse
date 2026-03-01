from django.urls import path
from copilot import views

app_name = "copilot"

urlpatterns = [
    # Pages
    path("", views.chat_page, name="chat"),
    path("generate/", views.generate_page, name="generate"),
    path("compare/", views.compare_page, name="compare"),
    path("about/", views.about_page, name="about"),
    path("pipeline/", views.pipeline_page, name="pipeline"),
    path("plugin/", views.plugin_page, name="plugin"),
    path("plugin/download/", views.plugin_download, name="plugin_download"),

    # AJAX API endpoints
    path("api/generate/", views.api_generate, name="api_generate"),
    path("api/score/", views.api_score, name="api_score"),
    path("api/self-correct/", views.api_self_correct, name="api_self_correct"),
    path("api/agentic/", views.api_agentic_generate, name="api_agentic"),

    # Pipeline API
    path("api/pipeline/status/", views.api_pipeline_status, name="api_pipeline_status"),
    path("api/pipeline/prompts/", views.api_pipeline_prompts, name="api_pipeline_prompts"),
    path("api/pipeline/prompt/<str:version>/", views.api_pipeline_prompt_detail, name="api_pipeline_prompt_detail"),

    # Studio Plugin API
    path("api/studio/send/", views.api_studio_send, name="studio_send"),
    path("api/studio/poll/", views.api_studio_poll, name="studio_poll"),
    path("api/studio/report/", views.api_studio_report, name="studio_report"),
    path("api/studio/heartbeat/", views.api_studio_heartbeat, name="studio_heartbeat"),
]
