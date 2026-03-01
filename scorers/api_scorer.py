import re
from scorers.compat import Scorer, op


KNOWN_SERVICES = {
    "Players", "Workspace", "ReplicatedStorage", "ServerStorage", "ServerScriptService",
    "StarterGui", "StarterPack", "StarterPlayer", "Lighting", "SoundService",
    "TweenService", "UserInputService", "RunService", "DataStoreService",
    "HttpService", "MarketplaceService", "PathfindingService", "PhysicsService",
    "CollectionService", "Chat", "Teams", "BadgeService", "TeleportService",
    "ProximityPromptService", "ContextActionService", "GuiService", "TextService",
    "PolicyService", "GroupService", "MessagingService", "MemoryStoreService",
    "LocalizationService", "InsertService", "GamePassService",
}

DEPRECATED_PATTERNS = [
    (r"\bwait\s*\(", r"\btask\.wait\s*\(", "Use task.wait() instead of wait()"),
    (r"\bspawn\s*\(", r"\btask\.spawn\s*\(", "Use task.spawn() instead of spawn()"),
    (r"\bdelay\s*\(", r"\btask\.delay\s*\(", "Use task.delay() instead of delay()"),
    (r"game\.Players\b", r'GetService\s*\(\s*"Players"\s*\)', "Use GetService('Players') instead of game.Players"),
    (r"game\.Workspace\b", r'GetService\s*\(\s*"Workspace"\s*\)', "Use GetService('Workspace') instead of game.Workspace"),
    (r"\.connect\s*\(", r"\.Connect\s*\(", "Use .Connect() not .connect() (capital C)"),
    (r'Instance\.new\s*\([^)]+,\s*\w+\)', None, "Don't pass parent as 2nd arg to Instance.new()"),
]

GOOD_PATTERNS = [
    (r"\.Connect\s*\(", "Uses .Connect()"),
    (r"Instance\.new\s*\(", "Uses Instance.new()"),
    (r"\btask\.\w+\s*\(", "Uses task library"),
    (r"CFrame", "Uses CFrame"),
    (r"Vector3", "Uses Vector3"),
    (r"GetService", "Uses GetService"),
]


class ApiScorer(Scorer):
    """Checks Roblox API correctness: GetService usage, deprecated patterns, good patterns."""

    @op()
    def score(self, output: str, **kwargs) -> dict:
        issues = []
        good_count = 0

        # Check GetService references against known services
        service_refs = re.findall(r'GetService\s*\(\s*"(\w+)"\s*\)', output)
        for svc in service_refs:
            if svc not in KNOWN_SERVICES:
                issues.append(f"Unknown service: '{svc}'")

        # Check deprecated patterns
        for deprecated_re, modern_re, msg in DEPRECATED_PATTERNS:
            if re.search(deprecated_re, output):
                # If there's a modern version, only flag if modern is absent
                if modern_re and re.search(modern_re, output):
                    continue
                issues.append(msg)

        # Count good patterns
        for pattern_re, _label in GOOD_PATTERNS:
            if re.search(pattern_re, output):
                good_count += 1

        # Score: deduct for issues, small bonus for good patterns
        deduction = len(issues) * 0.15
        bonus = min(good_count * 0.05, 0.2)
        score = max(0.0, min(1.0, 1.0 - deduction + bonus))

        return {
            "api_correct": len(issues) == 0,
            "api_score": round(score, 3),
            "api_issues": issues,
        }
