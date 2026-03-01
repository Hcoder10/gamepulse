from django.urls import path
from core import views

app_name = "core"

urlpatterns = [
    path("", views.landing, name="landing"),
    path("setup/", views.setup, name="setup"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("logout/", views.api_logout, name="logout"),

    # API — data
    path("api/game/", views.api_game_lookup, name="api_game"),
    path("api/portfolio/", views.api_portfolio, name="api_portfolio"),
    path("api/add-game/", views.api_add_game, name="api_add_game"),

    # API — AI features
    path("api/insights/", views.api_insights, name="api_insights"),
    path("api/briefing/", views.api_portfolio_briefing, name="api_briefing"),
    path("api/competitors/", views.api_competitor_analysis, name="api_competitors"),
    path("api/game-loop/", views.api_game_loop, name="api_game_loop"),
    path("api/trends/", views.api_trends, name="api_trends"),
    path("api/chat/", views.api_chat, name="api_chat"),
    path("api/social/", views.api_social_posts, name="api_social"),
    path("api/roadmap/", views.api_roadmap, name="api_roadmap"),
    path("api/revenue/", views.api_revenue, name="api_revenue"),
    path("api/description/", views.api_description, name="api_description"),
]
