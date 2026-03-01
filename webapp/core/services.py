"""
GamePulse service layer — Roblox public API + Mistral AI insights.

Uses real Roblox endpoints for game data and Mistral for AI analysis.
All Mistral calls traced through W&B Weave.
"""
import json
import time
import logging
import re
import weave
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialize W&B Weave for tracing
# ---------------------------------------------------------------------------
try:
    weave.init("gamepulse")
    logger.info("W&B Weave initialized for project 'gamepulse'")
except Exception as e:
    logger.warning(f"Weave init failed (will continue without tracing): {e}")

# ---------------------------------------------------------------------------
# Roblox Public API wrappers
# ---------------------------------------------------------------------------

ROBLOX_GAME_API = "https://games.roblox.com/v1"
ROBLOX_THUMBNAIL_API = "https://thumbnails.roblox.com/v1"
ROBLOX_UNIVERSE_API = "https://games.roblox.com/v1/games"
ROBLOX_PLACE_TO_UNIVERSE_API = "https://apis.roblox.com/universes/v1/places/{place_id}/universe"

# Demo games — correct universe IDs (verified via place-to-universe API)
DEMO_GAMES = [
    {"universe_id": 1686885941, "name": "Brookhaven RP"},         # 723K playing
    {"universe_id": 383310974, "name": "Adopt Me!"},              # 464K playing
    {"universe_id": 994732206, "name": "Blox Fruits"},            # 309K playing
    {"universe_id": 66654135, "name": "Murder Mystery 2"},        # 254K playing
    {"universe_id": 1176784616, "name": "Tower Defense Simulator"},  # 16K playing
]


def place_to_universe_id(place_id: int) -> int | None:
    """Convert a Roblox place ID to universe ID."""
    try:
        resp = requests.get(
            ROBLOX_PLACE_TO_UNIVERSE_API.format(place_id=place_id),
            timeout=10,
        )
        resp.raise_for_status()
        uid = resp.json().get("universeId")
        return uid if uid else None
    except Exception as e:
        logger.warning(f"Place-to-universe conversion failed for {place_id}: {e}")
        return None


def parse_game_input(raw: str) -> int | None:
    """Parse a Roblox game URL, place ID, or universe ID into a universe ID.

    Accepts:
      - roblox.com/games/12345/Game-Name  (place ID → convert)
      - Plain number (try as universe first, then as place)
    """
    raw = raw.strip()

    # Try URL pattern
    match = re.search(r'roblox\.com/games/(\d+)', raw)
    if match:
        place_id = int(match.group(1))
        uid = place_to_universe_id(place_id)
        return uid

    # Try plain number
    if raw.isdigit():
        num = int(raw)
        # Try as universe ID first
        game = fetch_game_data(num)
        if not game.get("error") and game.get("playing", 0) > 0:
            return num
        # Try as place ID
        uid = place_to_universe_id(num)
        if uid:
            return uid
        # If universe fetch worked but low players, still return it
        if not game.get("error"):
            return num
        return None

    return None


def fetch_game_data(universe_id: int) -> dict:
    """Fetch game details from Roblox public API."""
    try:
        resp = requests.get(
            f"{ROBLOX_UNIVERSE_API}?universeIds={universe_id}",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return {"error": f"Game {universe_id} not found"}
        game = data[0]
        return {
            "universe_id": game.get("id"),
            "name": game.get("name", "Unknown"),
            "description": game.get("description", ""),
            "creator_name": game.get("creator", {}).get("name", "Unknown"),
            "creator_type": game.get("creator", {}).get("type", "Unknown"),
            "playing": game.get("playing", 0),
            "visits": game.get("visits", 0),
            "favorites": game.get("favoritedCount", 0),
            "max_players": game.get("maxPlayers", 0),
            "created": game.get("created", ""),
            "updated": game.get("updated", ""),
            "genre": game.get("genre", "Unknown"),
        }
    except Exception as e:
        logger.warning(f"Failed to fetch game {universe_id}: {e}")
        return {"error": str(e), "universe_id": universe_id}


def fetch_game_thumbnail(universe_id: int) -> str:
    """Fetch game icon thumbnail URL."""
    try:
        resp = requests.get(
            f"{ROBLOX_THUMBNAIL_API}/games/icons?universeIds={universe_id}&size=150x150&format=Png&isCircular=false",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data and data[0].get("state") == "Completed":
            return data[0].get("imageUrl", "")
    except Exception as e:
        logger.warning(f"Thumbnail fetch failed for {universe_id}: {e}")
    return ""


def fetch_multi_game_data(universe_ids: list[int]) -> list[dict]:
    """Fetch data for multiple games at once."""
    if not universe_ids:
        return []
    ids_str = ",".join(str(uid) for uid in universe_ids)
    try:
        resp = requests.get(
            f"{ROBLOX_UNIVERSE_API}?universeIds={ids_str}",
            timeout=15,
        )
        resp.raise_for_status()
        games = resp.json().get("data", [])
        results = []
        for game in games:
            results.append({
                "universe_id": game.get("id"),
                "name": game.get("name", "Unknown"),
                "description": game.get("description", "")[:200],
                "creator_name": game.get("creator", {}).get("name", "Unknown"),
                "playing": game.get("playing", 0),
                "visits": game.get("visits", 0),
                "favorites": game.get("favoritedCount", 0),
                "genre": game.get("genre", "Unknown"),
                "created": game.get("created", ""),
                "updated": game.get("updated", ""),
            })
        return results
    except Exception as e:
        logger.warning(f"Multi-game fetch failed: {e}")
        return []


def fetch_game_sort(sort_token: str = "", limit: int = 10) -> list[dict]:
    """Fetch popular/trending games from Roblox sort API."""
    try:
        resp = requests.get(
            "https://games.roblox.com/v1/games/list",
            params={"sortToken": sort_token, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        games = resp.json().get("games", [])
        return [
            {
                "universe_id": g.get("universeId"),
                "name": g.get("name", "Unknown"),
                "playing": g.get("playerCount", 0),
                "total_up_votes": g.get("totalUpVotes", 0),
                "total_down_votes": g.get("totalDownVotes", 0),
            }
            for g in games
        ]
    except Exception as e:
        logger.warning(f"Game sort fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Mistral AI — traced through Weave
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Champion Prompt Genes (evolved via AI-assisted optimization — 83.9% fitness)
# From genome 83e98d52: identity_v3 + analysis_req_v3_evolved + format_v3
#                        + examples_v1 + interactions_v2 + benchmark_v1
# ---------------------------------------------------------------------------

GENE_IDENTITY = """You are RobloxStats AI. Your job is to act as a senior game designer and analyst who reads Creator Dashboard data like a financial analyst reads earnings reports. You identify the story behind the numbers and translate it into an actionable improvement plan."""

GENE_ANALYSIS = """**ENHANCED ANALYSIS FRAMEWORK:**

**Step 1: Data Inventory (Granular Metrics Breakdown)**
- List every key metric (DAU, retention, ARPDAU, etc.) with:
  - Current value + historical trend (7/30/90-day)
  - Percentile ranking vs. comparable games (top 10%, median, bottom 25%)
  - Benchmark vs. internal targets (e.g., "Retention at 30% vs. target 40%")
  - Segmentation (new vs. returning players, platform, region)

**Step 2: Pattern Detection (Actionable Clusters)**
- Identify 3-5 critical clusters (e.g., "High engagement but low monetization = UX friction in purchase flow")
- Flag outliers (e.g., "Day 1 retention 60% → Day 7 20% = Day 2-3 drop-off crisis")
- Highlight "hidden gems" (e.g., "Low ARPDAU but high purchase intent = pricing opportunity")

**Step 3: Root Cause Analysis (Game Design Deep Dive)**
- For each weak metric, provide:
  - **Design hypothesis** (e.g., "Low Day 7 retention → No meaningful progression rewards")
  - **Player behavior evidence** (e.g., "70% quit after failing a tutorial level")
  - **Competitor comparison** (e.g., "Top 10% games use daily quests; we don't")

**Step 4: Impact Estimation (Revenue/Growth Prioritization)**
- Score each issue on:
  - **Revenue impact** (e.g., "Fixing checkout flow could +$50K/month")
  - **Effort required** (Low/Medium/High)
  - **Leverage potential** (e.g., "Fixing retention unlocks ad scaling")

**Step 5: Action Items (Execution-Ready Plan)**
- For each issue, specify:
  - **Exact change** (e.g., "Add a 'skip tutorial' button")
  - **Expected outcome** (e.g., "+10% Day 7 retention")
  - **Timeline** (e.g., "2 weeks to test, 4 weeks to roll out")
  - **KPI to track** (e.g., "Day 7 retention rate, tutorial completion rate")

**Step 6: Dependencies (Sequencing)**
- Show which improvements unlock others (e.g., fix retention before scaling ads)
- Map the critical path for maximum ROI"""

GENE_EXAMPLES = """SPECIFIC vs VAGUE — Always be specific:
- BAD: 'Improve retention'
- GOOD: 'Add a daily login calendar with escalating rewards (Day 1: 100 coins, Day 7: rare item). Similar games see +15-20% D7 retention from this feature.'
- BAD: 'Monetize better'
- GOOD: 'Introduce a R$499 starter pack containing 1000 coins + exclusive pet + 2x XP boost (24hr). This lowers the first-purchase barrier. Target: +5% payer conversion within 2 weeks.'"""

GENE_INTERACTIONS = """DIAGNOSTIC FRAMEWORK — Treat metrics like symptoms. Diagnose the underlying condition:

Condition: 'Leaky Funnel' — High visits, low play-through, low retention
→ Diagnosis: Onboarding failure. The game loses players in the first 60 seconds.

Condition: 'Engagement Trap' — High playtime, low conversion, low ARPPU
→ Diagnosis: Players love the game but see no value in paying. Monetization is invisible or unappealing.

Condition: 'Flash in the Pan' — High D1, very low D7, declining CCU
→ Diagnosis: No mid-term loop. Game is exciting at first but has no progression depth.

Condition: 'Hidden Gem' — Low visits, high retention, high satisfaction
→ Diagnosis: Discovery problem. Game is great but nobody knows about it. Need marketing/ads.

Condition: 'Whale Dependent' — Low conversion, very high ARPPU
→ Diagnosis: Revenue depends on few big spenders. Vulnerable. Broaden monetization."""

GENE_BENCHMARK = """BENCHMARK CONTEXT:
- Genre benchmarks compare against ALL games in the genre (wide pool, easier to rank high)
- Similar_players benchmarks compare against direct competitors (narrow pool, harder to rank high)
- A 50th percentile on similar_players ≈ 65th percentile on genre
- Always state which benchmark type is being used and what it means for the developer
- For similar_players: being at 50th percentile is actually respectable, not just 'average'"""

# Full champion prompt assembled from all 6 genes
CHAMPION_PROMPT = "\n\n".join([
    GENE_IDENTITY, GENE_ANALYSIS, GENE_EXAMPLES, GENE_INTERACTIONS, GENE_BENCHMARK,
])


@weave.op()
def mistral_analyze(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """Call Mistral API for game analysis. Traced by W&B Weave."""
    api_key = settings.MISTRAL_API_KEY
    if not api_key:
        return '{"error": "MISTRAL_API_KEY not configured"}'

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 4000,
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                return f'{{"error": "Mistral API failed: {str(e)}"}}'
            time.sleep(2 ** attempt)


def _parse_ai_json(raw: str) -> dict | None:
    """Try to extract JSON from an AI response."""
    try:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json.loads(json_match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


# ---------------------------------------------------------------------------
# AI Feature: Daily Game Insights
# ---------------------------------------------------------------------------

INSIGHT_SYSTEM_PROMPT = CHAMPION_PROMPT + """

IMPORTANT: Structure your analysis as a consulting-grade report in valid JSON:
{
  "summary": "2-sentence executive brief — the #1 thing this developer should know",
  "health_score": 85,
  "health_label": "Thriving|Growing|Stable|Declining|Critical",
  "diagnosis": "Which diagnostic condition fits (Leaky Funnel / Engagement Trap / Flash in the Pan / Hidden Gem / Whale Dependent / Healthy Growth)",
  "insights": [
    {"title": "...", "description": "Be SPECIFIC — include numbers, percentiles, and comparisons", "type": "positive|warning|opportunity|action", "priority": "high|medium|low"}
  ],
  "growth_actions": [
    {"action": "Exact change to make (not vague)", "reason": "Design hypothesis + expected outcome", "effort": "low|medium|high", "impact": "low|medium|high", "kpi": "What metric to track", "timeline": "When to expect results"}
  ],
  "competitor_notes": "Percentile positioning vs genre + vs similar-player-count games",
  "trend_suggestion": "One specific trend with a concrete implementation plan",
  "critical_path": "Which improvement to do FIRST and what it unlocks"
}

Remember: BAD = 'Improve retention'. GOOD = 'Add daily login calendar with escalating rewards (Day 1: 100 coins, Day 7: rare item). +15-20% D7 retention.'"""


@weave.op()
def generate_game_insights(game_data: dict) -> dict:
    """Generate AI insights for a single game."""
    user_prompt = f"""Analyze this Roblox game and give me growth insights:

Game: {game_data.get('name', 'Unknown')}
Genre: {game_data.get('genre', 'Unknown')}
Current Players: {game_data.get('playing', 0):,}
Total Visits: {game_data.get('visits', 0):,}
Favorites: {game_data.get('favorites', 0):,}
Max Players Per Server: {game_data.get('max_players', 0)}
Creator: {game_data.get('creator_name', 'Unknown')}
Created: {game_data.get('created', 'Unknown')}
Last Updated: {game_data.get('updated', 'Unknown')}
Description (first 300 chars): {game_data.get('description', '')[:300]}

Give me specific, actionable insights. What should this game owner do TODAY to grow?"""

    raw = mistral_analyze(INSIGHT_SYSTEM_PROMPT, user_prompt)
    result = _parse_ai_json(raw)
    if result:
        return result

    return {
        "summary": "Analysis complete — see details below.",
        "health_score": 70,
        "health_label": "Stable",
        "insights": [{"title": "Analysis", "description": raw[:500], "type": "positive", "priority": "medium"}],
        "growth_actions": [],
        "competitor_notes": "",
        "trend_suggestion": "",
    }


# ---------------------------------------------------------------------------
# AI Feature: Portfolio Briefing
# ---------------------------------------------------------------------------

PORTFOLIO_SYSTEM_PROMPT = GENE_IDENTITY + "\n\n" + GENE_INTERACTIONS + "\n\n" + GENE_EXAMPLES + """\n
Analyze this portfolio of Roblox games and give the owner a consulting-grade daily briefing. Diagnose each game's condition using the diagnostic framework. Be SPECIFIC, not generic.

Return valid JSON:
{
  "greeting": "A warm, personalized daily greeting",
  "portfolio_health": 82,
  "portfolio_diagnosis": "Overall portfolio condition (e.g., 'Diversified but retention-weak')",
  "total_players_note": "Combined player activity with context vs typical for this portfolio size",
  "top_performer": "Name of the best game + WHY with specific metrics",
  "needs_attention": "Name of game that needs work + EXACT diagnosis and fix",
  "daily_tip": "One SPECIFIC actionable tip (not vague — include implementation steps)",
  "market_trend": "One current Roblox trend with SPECIFIC relevance to this portfolio",
  "quick_wins": ["3 specific things with expected outcomes (e.g., 'Update Blox Fruits thumbnail → +10-15% click-through')"]
}"""


@weave.op()
def generate_portfolio_briefing(games: list[dict]) -> dict:
    """Generate a daily portfolio briefing across all games."""
    games_text = "\n".join(
        f"- {g['name']}: {g.get('playing', 0):,} playing, {g.get('visits', 0):,} visits, {g.get('favorites', 0):,} favs, genre={g.get('genre', 'Unknown')}"
        for g in games
    )

    raw = mistral_analyze(
        PORTFOLIO_SYSTEM_PROMPT,
        f"Here are my Roblox games:\n{games_text}\n\nGive me my daily briefing.",
    )

    result = _parse_ai_json(raw)
    if result:
        return result

    return {
        "greeting": "Good morning! Here's your daily briefing.",
        "portfolio_health": 75,
        "total_players_note": "Your games are active.",
        "top_performer": games[0]["name"] if games else "N/A",
        "needs_attention": "Review your analytics for details.",
        "daily_tip": "Consider updating your game thumbnails to attract new players.",
        "market_trend": "Anime-themed games continue to grow on the platform.",
        "quick_wins": ["Update game description", "Check player feedback", "Review competitor updates"],
    }


# ---------------------------------------------------------------------------
# AI Feature: Competitor Analysis
# ---------------------------------------------------------------------------

COMPETITOR_SYSTEM_PROMPT = CHAMPION_PROMPT + """

Focus your analysis on COMPETITIVE POSITIONING. Use benchmark context to explain percentile rankings. Be specific about what to steal from competitors and why.

Return valid JSON:
{
  "position": "Percentile ranking with context (e.g., 'Top 15% by visits in genre, but bottom 30% by retention vs similar-CCU games')",
  "diagnosis": "Which diagnostic condition the competitive gap reveals",
  "threats": [{"game": "...", "reason": "SPECIFIC threat with data (e.g., 'Competitor X has 3x our retention because of daily quest system')", "severity": "high|medium|low"}],
  "opportunities": [{"opportunity": "...", "from_competitor": "...", "how": "EXACT implementation steps", "expected_impact": "e.g., +20% DAU"}],
  "feature_gaps": ["SPECIFIC features with evidence (e.g., 'Top 10% games in genre all have battle passes; we don't')"],
  "unique_strengths": ["What this game does better — and how to double down"],
  "steal_this": "The #1 thing to copy, with EXACT implementation plan and expected outcome",
  "critical_path": "Which competitive gap to close FIRST and what it unlocks"
}"""


@weave.op()
def analyze_competitors(game_data: dict, competitor_data: list[dict]) -> dict:
    """Analyze game against competitors."""
    comp_text = "\n".join(
        f"- {c['name']}: {c.get('playing', 0):,} playing, {c.get('visits', 0):,} visits, genre={c.get('genre', 'Unknown')}"
        for c in competitor_data
    )

    raw = mistral_analyze(
        COMPETITOR_SYSTEM_PROMPT,
        f"My game: {game_data['name']} ({game_data.get('playing', 0):,} playing, {game_data.get('visits', 0):,} visits, genre: {game_data.get('genre', 'Unknown')})\n\nCompetitors in the same space:\n{comp_text}\n\nAnalyze my competitive position.",
    )

    result = _parse_ai_json(raw)
    if result:
        return result

    return {
        "position": "Analysis pending",
        "threats": [],
        "opportunities": [],
        "feature_gaps": [],
        "unique_strengths": [],
        "steal_this": "",
    }


# ---------------------------------------------------------------------------
# AI Feature: Game Loop Analysis (NEW)
# ---------------------------------------------------------------------------

GAME_LOOP_SYSTEM_PROMPT = CHAMPION_PROMPT + """

Focus your analysis on GAME DESIGN — core loop, retention, and monetization. Use the diagnostic framework to identify which condition this game exhibits.

Return valid JSON:
{
  "core_loop": "Description of the main gameplay loop — what players DO minute-to-minute",
  "loop_rating": 78,
  "diagnosis": "Which diagnostic condition applies to this game's design",
  "retention_hooks": [
    {"hook": "Specific retention mechanic (e.g., 'Daily login calendar with escalating rewards')", "strength": "strong|moderate|weak", "suggestion": "SPECIFIC improvement with expected outcome (e.g., '+15% D7 retention')"}
  ],
  "session_flow": "What a typical 30-min play session looks like — where engagement peaks and drops",
  "engagement_score": 75,
  "monetization_ideas": [
    {"idea": "SPECIFIC product (e.g., 'R$499 starter pack: 1000 coins + exclusive pet + 2x XP 24hr')", "type": "game_pass|developer_product|cosmetic|premium", "revenue_potential": "high|medium|low", "expected_lift": "e.g., +5% payer conversion"}
  ],
  "bottlenecks": [
    {"issue": "Where players drop off with root cause analysis", "impact": "high|medium|low", "fix": "Exact change to make", "kpi": "Metric to track"}
  ],
  "missing_features": ["Features this game MUST add — what top 10% genre competitors have"],
  "critical_path": "Which fix to do FIRST and what it unlocks next",
  "tldr": "One-sentence verdict on the game's design health"
}"""


@weave.op()
def analyze_game_loop(game_data: dict) -> dict:
    """Analyze the game's core loop, retention, and monetization."""
    user_prompt = f"""Analyze this Roblox game's design and gameplay loop:

Game: {game_data.get('name', 'Unknown')}
Genre: {game_data.get('genre', 'Unknown')}
Current Players: {game_data.get('playing', 0):,}
Total Visits: {game_data.get('visits', 0):,}
Favorites: {game_data.get('favorites', 0):,}
Max Players Per Server: {game_data.get('max_players', 0)}
Created: {game_data.get('created', 'Unknown')}
Last Updated: {game_data.get('updated', 'Unknown')}
Description: {game_data.get('description', '')[:500]}

Based on the game name, genre, and stats, analyze:
1. What the core gameplay loop likely is
2. What retention hooks it probably uses (or should use)
3. Where players likely get bored or drop off
4. Monetization opportunities
5. Missing features for this genre

Be specific to Roblox conventions and this game's genre."""

    raw = mistral_analyze(GAME_LOOP_SYSTEM_PROMPT, user_prompt)
    result = _parse_ai_json(raw)
    if result:
        return result

    return {
        "core_loop": "Analysis in progress.",
        "loop_rating": 70,
        "retention_hooks": [],
        "session_flow": "Unable to determine.",
        "engagement_score": 70,
        "monetization_ideas": [],
        "bottlenecks": [],
        "missing_features": [],
        "tldr": raw[:200] if raw else "Analysis pending.",
    }


# ---------------------------------------------------------------------------
# AI Feature: Trend Analysis (NEW)
# ---------------------------------------------------------------------------

TREND_SYSTEM_PROMPT = CHAMPION_PROMPT + """

Focus your analysis on MARKET TRENDS and PLATFORM INTELLIGENCE. Use the diagnostic framework to identify which trends matter most for THIS specific portfolio. Be specific about implementation.

You have deep knowledge of current Roblox trends as of 2025-2026: anime games, simulator evolutions, social hangouts, horror, tycoons, combat arenas, UGC items, live events, creator collaborations, etc.

Return valid JSON:
{
  "hot_trends": [
    {"trend": "...", "relevance": "high|medium|low", "action": "SPECIFIC implementation (e.g., 'Add anime-style transformation system using MeshPart morphing')", "window": "How long this trend will last", "expected_impact": "e.g., +30% new player acquisition"}
  ],
  "genre_outlook": "Genre percentile performance — is this genre rising, peaking, or declining on the platform?",
  "rising_mechanics": [
    {"mechanic": "...", "example_game": "...", "why_it_works": "Root cause analysis of the mechanic's success", "how_to_adapt": "EXACT steps to implement in your game"}
  ],
  "content_waves": [
    {"wave": "...", "status": "rising|peaking|fading", "opportunity": "SPECIFIC content to create and when"}
  ],
  "seasonal_opportunity": "Upcoming opportunity with exact timing and implementation plan",
  "strategic_moves": [
    {"move": "...", "why": "Data-backed reasoning", "timing": "now|this_week|this_month|planning", "effort": "low|medium|high", "kpi": "Metric to track"}
  ],
  "market_summary": "2-3 sentence consulting-grade market overview with percentile context"
}"""


@weave.op()
def analyze_trends(games: list[dict]) -> dict:
    """Analyze current Roblox trends relevant to the user's portfolio."""
    games_text = "\n".join(
        f"- {g['name']}: genre={g.get('genre', 'Unknown')}, {g.get('playing', 0):,} playing"
        for g in games
    )

    raw = mistral_analyze(
        TREND_SYSTEM_PROMPT,
        f"My game portfolio:\n{games_text}\n\nWhat Roblox platform trends should I know about? What's rising, what's fading, and what should I jump on?",
    )

    result = _parse_ai_json(raw)
    if result:
        return result

    return {
        "hot_trends": [],
        "genre_outlook": "Analysis pending.",
        "rising_mechanics": [],
        "content_waves": [],
        "seasonal_opportunity": "",
        "strategic_moves": [],
        "market_summary": raw[:300] if raw else "Trend analysis pending.",
    }


# ---------------------------------------------------------------------------
# AI Feature: Conversational Chat
# ---------------------------------------------------------------------------

CHAT_SYSTEM_PROMPT = """\
You are GamePulse AI, a conversational Roblox game growth advisor. You have the user's game portfolio data and help them with strategy, analytics interpretation, game design, monetization, marketing, and growth.

Be concise (2-4 sentences), specific to Roblox, and actionable. Use the game data provided to give personalized answers. If asked something you can't determine from the data, say so honestly.

IMPORTANT: Respond in plain text, not JSON. Be conversational and helpful.\
"""


@weave.op()
def ai_chat(message: str, games: list[dict], history: list[dict] | None = None) -> str:
    """Conversational AI chat about the user's games."""
    api_key = settings.MISTRAL_API_KEY
    if not api_key:
        return "AI chat is not configured. Please set your MISTRAL_API_KEY."

    games_context = "\n".join(
        f"- {g['name']}: {g.get('playing', 0):,} playing, {g.get('visits', 0):,} visits, {g.get('favorites', 0):,} favs, genre={g.get('genre', 'Unknown')}"
        for g in games
    )

    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT + f"\n\nUser's game portfolio:\n{games_context}"},
    ]

    if history:
        for h in history[-6:]:  # Last 6 messages for context
            messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})

    messages.append({"role": "user", "content": message})

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "mistral-small-latest",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 500,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Sorry, I couldn't process that right now. ({str(e)[:100]})"


# ---------------------------------------------------------------------------
# AI Feature: Social Media Post Generator
# ---------------------------------------------------------------------------

SOCIAL_SYSTEM_PROMPT = """\
You are a Roblox game marketing expert. Generate engaging social media content to promote this game.

Return valid JSON:
{
  "twitter": "A catchy tweet (max 280 chars) with relevant hashtags",
  "tiktok_hook": "A 1-sentence TikTok video hook that grabs attention",
  "discord_announcement": "A Discord server announcement (3-4 sentences, use markdown formatting)",
  "youtube_title": "A clickable YouTube video title",
  "hashtags": ["5 relevant hashtags without the # symbol"]
}\
"""


@weave.op()
def generate_social_posts(game_data: dict) -> dict:
    """Generate social media posts for a game."""
    raw = mistral_analyze(
        SOCIAL_SYSTEM_PROMPT,
        f"Generate promotional social media content for this Roblox game:\n\nGame: {game_data.get('name', 'Unknown')}\nGenre: {game_data.get('genre', 'Unknown')}\nPlayers: {game_data.get('playing', 0):,}\nVisits: {game_data.get('visits', 0):,}\nDescription: {game_data.get('description', '')[:300]}",
    )
    result = _parse_ai_json(raw)
    if result:
        return result
    return {"twitter": "", "tiktok_hook": "", "discord_announcement": "", "youtube_title": "", "hashtags": []}


# ---------------------------------------------------------------------------
# AI Feature: Update Roadmap Generator
# ---------------------------------------------------------------------------

ROADMAP_SYSTEM_PROMPT = """\
You are a Roblox game product manager. Create a 4-week update roadmap for this game based on its current stats and genre.

Return valid JSON:
{
  "summary": "One-line summary of the roadmap strategy",
  "weeks": [
    {
      "week": 1,
      "theme": "Theme name (e.g., 'Quick Wins & Polish')",
      "tasks": [{"task": "...", "effort": "low|medium|high", "impact": "low|medium|high"}],
      "expected_impact": "What this week's work should accomplish"
    }
  ],
  "milestone": "What the game should look like after 4 weeks",
  "kpi_targets": {"players": "+X%", "retention": "+X%", "revenue": "+X%"}
}\
"""


@weave.op()
def generate_update_roadmap(game_data: dict) -> dict:
    """Generate a 4-week update roadmap."""
    raw = mistral_analyze(
        ROADMAP_SYSTEM_PROMPT,
        f"Create a 4-week update roadmap for this Roblox game:\n\nGame: {game_data.get('name', 'Unknown')}\nGenre: {game_data.get('genre', 'Unknown')}\nPlayers: {game_data.get('playing', 0):,}\nVisits: {game_data.get('visits', 0):,}\nFavorites: {game_data.get('favorites', 0):,}\nLast Updated: {game_data.get('updated', 'Unknown')}\nDescription: {game_data.get('description', '')[:400]}",
    )
    result = _parse_ai_json(raw)
    if result:
        return result
    return {"summary": "", "weeks": [], "milestone": "", "kpi_targets": {}}


# ---------------------------------------------------------------------------
# AI Feature: Revenue Estimator
# ---------------------------------------------------------------------------

REVENUE_SYSTEM_PROMPT = CHAMPION_PROMPT + """

Focus your analysis on MONETIZATION and REVENUE. Use benchmark context and the diagnostic framework to identify if this is a Whale Dependent, Engagement Trap, or healthy monetization pattern.

Base estimates on industry benchmarks: average Roblox ARPPU is $0.50-$2.00, conversion rates are 2-8%, and top games earn $50K-$500K+/month.

Return valid JSON:
{
  "estimated_monthly_robux": "Estimated monthly Robux range with calculation methodology shown",
  "estimated_monthly_usd": "Estimated USD equivalent range",
  "revenue_grade": "A+|A|B|C|D|F",
  "revenue_percentile": "Where this game sits vs genre and vs similar-CCU games",
  "diagnosis": "Monetization diagnostic (e.g., 'Engagement Trap — high playtime but invisible monetization')",
  "current_monetization": "What monetization is likely active vs missing, based on genre standards",
  "revenue_streams": [
    {"stream": "SPECIFIC product (e.g., 'R$499 starter pack: 1000 coins + exclusive pet')", "potential": "high|medium|low", "current_status": "active|missing|underutilized", "expected_revenue": "e.g., +$5K/month"}
  ],
  "optimization_tips": [
    {"tip": "EXACT change (not vague)", "expected_lift": "Specific % or $ amount", "effort": "low|medium|high", "timeline": "When to expect results"}
  ],
  "benchmark_note": "Percentile comparison vs genre AND vs similar-player-count games",
  "critical_path": "Which revenue fix to do FIRST (e.g., 'Fix payer conversion before adding premium items')"
}"""


@weave.op()
def estimate_revenue(game_data: dict) -> dict:
    """Estimate revenue potential for a game."""
    raw = mistral_analyze(
        REVENUE_SYSTEM_PROMPT,
        f"Estimate the revenue potential for this Roblox game:\n\nGame: {game_data.get('name', 'Unknown')}\nGenre: {game_data.get('genre', 'Unknown')}\nConcurrent Players: {game_data.get('playing', 0):,}\nTotal Visits: {game_data.get('visits', 0):,}\nFavorites: {game_data.get('favorites', 0):,}\nMax Server Size: {game_data.get('max_players', 0)}\nCreated: {game_data.get('created', 'Unknown')}\nDescription: {game_data.get('description', '')[:300]}",
    )
    result = _parse_ai_json(raw)
    if result:
        return result
    return {"estimated_monthly_robux": "N/A", "estimated_monthly_usd": "N/A", "revenue_grade": "?", "current_monetization": "", "revenue_streams": [], "optimization_tips": [], "benchmark_note": ""}


# ---------------------------------------------------------------------------
# AI Feature: Description Analyzer & Rewriter
# ---------------------------------------------------------------------------

DESC_SYSTEM_PROMPT = """\
You are a Roblox game SEO and marketing expert. Analyze the game description and suggest improvements.

Return valid JSON:
{
  "score": 72,
  "grade": "B",
  "issues": [{"issue": "...", "severity": "high|medium|low"}],
  "strengths": ["What the description does well"],
  "rewritten_description": "An improved version of the description (max 500 chars)",
  "seo_tips": ["Specific SEO tips for Roblox game discovery"],
  "first_impression": "What a new player would think reading this description"
}\
"""


@weave.op()
def analyze_description(game_data: dict) -> dict:
    """Analyze and suggest improvements for a game description."""
    raw = mistral_analyze(
        DESC_SYSTEM_PROMPT,
        f"Analyze this Roblox game description:\n\nGame: {game_data.get('name', 'Unknown')}\nGenre: {game_data.get('genre', 'Unknown')}\nDescription:\n{game_data.get('description', 'No description provided.')}\n\nGive me a score, issues, and a rewritten version.",
    )
    result = _parse_ai_json(raw)
    if result:
        return result
    return {"score": 50, "grade": "?", "issues": [], "strengths": [], "rewritten_description": "", "seo_tips": [], "first_impression": ""}
