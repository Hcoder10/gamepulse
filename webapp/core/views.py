"""GamePulse views — pages and API endpoints."""
import json
import traceback

from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from core import services


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def landing(request):
    return render(request, "core/landing.html")


def setup(request):
    """Onboarding — enter username and games."""
    if request.method == "POST":
        username = request.POST.get("username", "").strip() or "Developer"
        game_inputs = request.POST.get("games", "").strip()
        use_demo = request.POST.get("use_demo") == "1"

        # Parse game inputs (URLs, place IDs, or universe IDs)
        game_ids = []
        if use_demo:
            game_ids = [g["universe_id"] for g in services.DEMO_GAMES]
        elif game_inputs:
            for line in game_inputs.replace(",", "\n").split("\n"):
                line = line.strip()
                if not line:
                    continue
                uid = services.parse_game_input(line)
                if uid:
                    game_ids.append(uid)

        request.session["username"] = username
        request.session["game_ids"] = game_ids
        return redirect("/dashboard/")

    return render(request, "core/setup.html", {
        "username": request.session.get("username", ""),
        "game_ids": request.session.get("game_ids", []),
    })


def dashboard(request):
    """Main dashboard — redirects to setup if no profile."""
    username = request.session.get("username")
    game_ids = request.session.get("game_ids", [])
    if not username:
        return redirect("/setup/")
    return render(request, "core/dashboard.html", {
        "username": username,
        "game_ids_json": json.dumps(game_ids),
    })


# ---------------------------------------------------------------------------
# API: Game data
# ---------------------------------------------------------------------------

@csrf_exempt
def api_game_lookup(request):
    """Look up a Roblox game by universe ID, place ID, or URL."""
    raw_input = request.GET.get("q", "") or request.GET.get("universe_id", "")
    if not raw_input:
        return JsonResponse({"error": "Game ID or URL required"}, status=400)

    # Try direct universe ID first
    try:
        uid = int(raw_input)
        game = services.fetch_game_data(uid)
        if not game.get("error"):
            game["thumbnail"] = services.fetch_game_thumbnail(uid)
            return JsonResponse(game)
    except ValueError:
        pass

    # Try parse as URL/place ID
    uid = services.parse_game_input(raw_input)
    if uid:
        game = services.fetch_game_data(uid)
        game["thumbnail"] = services.fetch_game_thumbnail(uid)
        return JsonResponse(game)

    return JsonResponse({"error": "Could not resolve game ID"}, status=400)


@csrf_exempt
def api_portfolio(request):
    """Return portfolio data for session games or demo games."""
    game_ids = request.session.get("game_ids", [])
    if not game_ids:
        game_ids = [g["universe_id"] for g in services.DEMO_GAMES]

    games = services.fetch_multi_game_data(game_ids)

    # Fetch thumbnails
    for g in games:
        g["thumbnail"] = services.fetch_game_thumbnail(g["universe_id"])

    return JsonResponse({"games": games})


@csrf_exempt
def api_add_game(request):
    """Add a game to the user's session portfolio."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        body = json.loads(request.body)
        raw_input = body.get("game_input", "")
        if not raw_input:
            return JsonResponse({"error": "game_input required"}, status=400)

        uid = services.parse_game_input(raw_input)
        if not uid:
            # Try direct as universe ID
            try:
                uid = int(raw_input)
            except ValueError:
                return JsonResponse({"error": "Could not resolve game ID"}, status=400)

        game = services.fetch_game_data(uid)
        if game.get("error"):
            return JsonResponse(game, status=400)

        # Add to session
        game_ids = request.session.get("game_ids", [])
        if uid not in game_ids:
            game_ids.append(uid)
            request.session["game_ids"] = game_ids

        game["thumbnail"] = services.fetch_game_thumbnail(uid)
        return JsonResponse({"game": game, "added": True})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ---------------------------------------------------------------------------
# API: AI insights
# ---------------------------------------------------------------------------

@csrf_exempt
@require_POST
def api_insights(request):
    """Generate AI insights for a game."""
    try:
        body = json.loads(request.body)
        universe_id = body.get("universe_id")
        if not universe_id:
            return JsonResponse({"error": "universe_id required"}, status=400)

        game_data = services.fetch_game_data(int(universe_id))
        if "error" in game_data:
            return JsonResponse(game_data, status=400)

        insights = services.generate_game_insights(game_data)
        return JsonResponse({"game": game_data, "insights": insights})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def api_portfolio_briefing(request):
    """Generate daily portfolio briefing."""
    try:
        body = json.loads(request.body)
        universe_ids = body.get("universe_ids", [])
        if not universe_ids:
            universe_ids = request.session.get("game_ids", [])
        if not universe_ids:
            universe_ids = [g["universe_id"] for g in services.DEMO_GAMES]

        games = services.fetch_multi_game_data([int(uid) for uid in universe_ids])
        briefing = services.generate_portfolio_briefing(games)
        return JsonResponse({"games": games, "briefing": briefing})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def api_competitor_analysis(request):
    """Analyze game against competitors."""
    try:
        body = json.loads(request.body)
        universe_id = body.get("universe_id")
        competitor_ids = body.get("competitor_ids", [])

        if not universe_id:
            return JsonResponse({"error": "universe_id required"}, status=400)

        game_data = services.fetch_game_data(int(universe_id))
        if "error" in game_data:
            return JsonResponse(game_data, status=400)

        if competitor_ids:
            competitors = services.fetch_multi_game_data([int(c) for c in competitor_ids])
        else:
            # Use other games in portfolio as competitors, or demo games
            session_ids = request.session.get("game_ids", [])
            other_ids = [uid for uid in session_ids if uid != int(universe_id)]
            if not other_ids:
                other_ids = [g["universe_id"] for g in services.DEMO_GAMES if g["universe_id"] != int(universe_id)]
            competitors = services.fetch_multi_game_data(other_ids[:5])

        analysis = services.analyze_competitors(game_data, competitors)
        return JsonResponse({"game": game_data, "competitors": competitors, "analysis": analysis})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def api_game_loop(request):
    """Analyze a game's core loop, retention, and monetization."""
    try:
        body = json.loads(request.body)
        universe_id = body.get("universe_id")
        if not universe_id:
            return JsonResponse({"error": "universe_id required"}, status=400)

        game_data = services.fetch_game_data(int(universe_id))
        if "error" in game_data:
            return JsonResponse(game_data, status=400)

        analysis = services.analyze_game_loop(game_data)
        return JsonResponse({"game": game_data, "analysis": analysis})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def api_trends(request):
    """Analyze current Roblox trends relevant to user's portfolio."""
    try:
        body = json.loads(request.body)
        universe_ids = body.get("universe_ids", [])
        if not universe_ids:
            universe_ids = request.session.get("game_ids", [])
        if not universe_ids:
            universe_ids = [g["universe_id"] for g in services.DEMO_GAMES]

        games = services.fetch_multi_game_data([int(uid) for uid in universe_ids])
        trends = services.analyze_trends(games)
        return JsonResponse({"games": games, "trends": trends})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def api_chat(request):
    """AI conversational chat about user's games."""
    try:
        body = json.loads(request.body)
        message = body.get("message", "")
        history = body.get("history", [])
        if not message:
            return JsonResponse({"error": "message required"}, status=400)

        universe_ids = request.session.get("game_ids", [])
        if not universe_ids:
            universe_ids = [g["universe_id"] for g in services.DEMO_GAMES]

        games = services.fetch_multi_game_data([int(uid) for uid in universe_ids])
        reply = services.ai_chat(message, games, history)
        return JsonResponse({"reply": reply})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def api_social_posts(request):
    """Generate social media posts for a game."""
    try:
        body = json.loads(request.body)
        universe_id = body.get("universe_id")
        if not universe_id:
            return JsonResponse({"error": "universe_id required"}, status=400)

        game_data = services.fetch_game_data(int(universe_id))
        if "error" in game_data:
            return JsonResponse(game_data, status=400)

        posts = services.generate_social_posts(game_data)
        return JsonResponse({"game": game_data, "posts": posts})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def api_roadmap(request):
    """Generate a 4-week update roadmap for a game."""
    try:
        body = json.loads(request.body)
        universe_id = body.get("universe_id")
        if not universe_id:
            return JsonResponse({"error": "universe_id required"}, status=400)

        game_data = services.fetch_game_data(int(universe_id))
        if "error" in game_data:
            return JsonResponse(game_data, status=400)

        roadmap = services.generate_update_roadmap(game_data)
        return JsonResponse({"game": game_data, "roadmap": roadmap})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def api_revenue(request):
    """Estimate revenue potential for a game."""
    try:
        body = json.loads(request.body)
        universe_id = body.get("universe_id")
        if not universe_id:
            return JsonResponse({"error": "universe_id required"}, status=400)

        game_data = services.fetch_game_data(int(universe_id))
        if "error" in game_data:
            return JsonResponse(game_data, status=400)

        revenue = services.estimate_revenue(game_data)
        return JsonResponse({"game": game_data, "revenue": revenue})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def api_description(request):
    """Analyze and improve a game's description."""
    try:
        body = json.loads(request.body)
        universe_id = body.get("universe_id")
        if not universe_id:
            return JsonResponse({"error": "universe_id required"}, status=400)

        game_data = services.fetch_game_data(int(universe_id))
        if "error" in game_data:
            return JsonResponse(game_data, status=400)

        analysis = services.analyze_description(game_data)
        return JsonResponse({"game": game_data, "analysis": analysis})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_logout(request):
    """Clear session and redirect to landing."""
    request.session.flush()
    return redirect("/")
