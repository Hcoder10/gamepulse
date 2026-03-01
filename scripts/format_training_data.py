"""Format raw Luau files into fine-tuning JSONL.

Usage:
    python scripts/format_training_data.py [--no-synthetic] [--max-samples N]

Takes raw Luau files from data/raw_luau/ and creates:
1. Task descriptions for each file (via Mistral or heuristic extraction)
2. JSONL training data with messages array: [{system, user, assistant}]
3. Optional synthetic hardening examples (edge cases, error handling)

Output: data/training_data.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MISTRAL_API_KEY, MISTRAL_MODEL

RAW_DIR = PROJECT_ROOT / "data" / "raw_luau"
OUTPUT_FILE = PROJECT_ROOT / "data" / "training_data.jsonl"

SYSTEM_PROMPT = """\
You are an expert Roblox Luau programmer. Generate complete, production-ready Luau scripts.

Structure every script like this:
-- Brief description of what this script does
local Services = game:GetService("ServiceName")
-- Configuration
-- Functions
-- Main logic

Rules:

1. Services: Access via game:GetService("Players"), never game.Players directly.
2. Variables: Declare everything with `local`. No globals.
3. Modern API only:
   - task.wait() / task.spawn() / task.delay() instead of wait() / spawn() / delay()
   - .Connect() with capital C, never .connect()
   - Instance.new("Part") then set .Parent separately, never Instance.new("Part", parent)
4. Safety patterns:
   - Wrap DataStore calls in pcall: local ok, result = pcall(function() return store:GetAsync(key) end)
   - Check FindFirstChild before accessing: local obj = parent:FindFirstChild("Name") if obj then ... end
   - Every while true do loop must contain task.wait() or another yield
   - Use WaitForChild() for objects that load asynchronously
5. Naming: PascalCase for services (local Players = ...), camelCase for variables and functions.
6. Comments: Start with a one-line description comment. Add a brief comment before each major section.
7. Completeness: Implement every feature in the task. No placeholders, no TODOs, no "add your code here". Every function must have a full body.
8. Cleanup: Store connections in variables. Disconnect when appropriate. Destroy unused instances.

Output only the Luau code. No markdown fences, no explanations.\
"""


def extract_task_description_heuristic(code: str, filename: str) -> str:
    """Extract a task description from code using heuristics (no API call needed)."""
    lines = code.strip().split("\n")

    # Try to get the first comment as description (skip author/license lines)
    first_comment = ""
    skip_words = [
        "author", "copyright", "license", "mit", "apache", "created by",
        "written by", "modified by", "version", "date:", "contributors",
        "@", "http", "www.", ".com", "github",
    ]
    for line in lines[:8]:
        stripped = line.strip()
        if stripped.startswith("--"):
            comment = stripped.lstrip("-").strip()
            if len(comment) > 15 and not comment.startswith("!"):
                # Skip author/license/metadata comments
                if any(w in comment.lower() for w in skip_words):
                    continue
                # Skip if it's just a name (no spaces or very short)
                if len(comment.split()) < 3:
                    continue
                first_comment = comment
                break

    # Detect what the script does by looking at patterns
    features = []
    if re.search(r"PlayerAdded|CharacterAdded", code):
        features.append("player handling")
    if re.search(r"DataStore|GetAsync|SetAsync", code):
        features.append("data persistence")
    if re.search(r"RemoteEvent|RemoteFunction", code):
        features.append("client-server communication")
    if re.search(r"TweenService|Tween", code):
        features.append("animations/tweening")
    if re.search(r"UserInputService|ContextActionService", code):
        features.append("input handling")
    if re.search(r"GUI|Frame|TextLabel|TextButton|ScreenGui", code):
        features.append("UI elements")
    if re.search(r"Humanoid|Character|Health", code):
        features.append("character mechanics")
    if re.search(r"CFrame|Vector3|Ray", code):
        features.append("physics/spatial math")
    if re.search(r"MarketplaceService|GamePassService|ProductId", code):
        features.append("monetization")
    if re.search(r"Chat|Message|TextChatService", code):
        features.append("chat system")
    if re.search(r"Sound|SoundService|PlaybackSpeed", code):
        features.append("audio")
    if re.search(r"Lighting|Atmosphere|Sky", code):
        features.append("lighting/atmosphere")
    if re.search(r"PathfindingService|MoveTo", code):
        features.append("NPC pathfinding")
    if re.search(r"CollectionService|Tagged", code):
        features.append("tag-based systems")
    if re.search(r"RunService.*Heartbeat|Stepped", code):
        features.append("per-frame updates")

    # Build description
    if first_comment and features:
        return f"Write a Roblox Luau script that implements {first_comment.lower()}. The script should handle {', '.join(features)}."
    elif first_comment:
        return f"Write a Roblox Luau script: {first_comment}"
    elif features:
        clean_name = re.sub(r"[_\-]", " ", filename.split("__")[-1].replace(".lua", "").replace(".luau", ""))
        return f"Write a Roblox Luau script for {clean_name} that handles {', '.join(features)}."
    else:
        clean_name = re.sub(r"[_\-]", " ", filename.split("__")[-1].replace(".lua", "").replace(".luau", ""))
        return f"Write a complete Roblox Luau script for: {clean_name}"


def generate_task_description_llm(code: str, filename: str) -> str:
    """Generate a task description using Mistral (higher quality but uses API)."""
    from mistralai import Mistral

    client = Mistral(api_key=MISTRAL_API_KEY)

    prompt = f"""Given this Roblox Luau script, write a concise task description (2-3 sentences) that someone would give to a programmer to produce this code. Focus on what the script does functionally, not implementation details.

Script:
```lua
{code[:2000]}
```

Respond with ONLY the task description, no quotes or formatting."""

    try:
        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        desc = response.choices[0].message.content.strip()
        if len(desc) > 20:
            return desc
    except Exception as e:
        print(f"    LLM description failed: {e}")

    return extract_task_description_heuristic(code, filename)


def modernize_code(code: str) -> str:
    """Apply basic modernization to training data (so model learns modern patterns)."""
    # Replace deprecated patterns
    code = re.sub(r"\bwait\(\)", "task.wait()", code)
    code = re.sub(r"\bwait\((\d+(?:\.\d+)?)\)", r"task.wait(\1)", code)
    code = re.sub(r"\bspawn\(", "task.spawn(", code)
    code = re.sub(r"\bdelay\(", "task.delay(", code)
    # Fix .connect -> .Connect (but not :Connect)
    code = re.sub(r"\.connect\(", ".Connect(", code)
    return code


def format_sample(task_description: str, code: str) -> dict:
    """Format a single training sample as a messages array."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_description},
            {"role": "assistant", "content": code},
        ]
    }


def process_raw_files(use_llm: bool = False, max_samples: int = 1000) -> list[dict]:
    """Process all raw Luau files into training samples."""
    samples = []

    # Collect all raw files
    all_files = []
    for subdir in ["github", "github_search", "stack_v2"]:
        src_dir = RAW_DIR / subdir
        if src_dir.exists():
            all_files.extend(src_dir.glob("*"))

    print(f"  Found {len(all_files)} raw files")

    for i, fpath in enumerate(all_files):
        if len(samples) >= max_samples:
            break

        if i % 50 == 0:
            print(f"  Processing {i}/{len(all_files)} (saved {len(samples)})...")

        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Modernize the code
        content = modernize_code(content)

        # Generate task description
        if use_llm and len(samples) % 5 == 0:
            # Use LLM for every 5th sample, heuristic for the rest
            task_desc = generate_task_description_llm(content, fpath.name)
            time.sleep(0.5)  # Rate limit
        else:
            task_desc = extract_task_description_heuristic(content, fpath.name)

        sample = format_sample(task_desc, content)
        samples.append(sample)

    return samples


def generate_synthetic_hardening(n_samples: int = 50) -> list[dict]:
    """Generate synthetic training examples for edge cases and error handling.

    These cover patterns that real code often misses but our scorers check for.
    """
    print(f"\n  Generating {n_samples} synthetic hardening examples...")

    # Pre-built synthetic examples covering common scorer failures
    synthetic_tasks = [
        {
            "task": "Write a DataStore script that saves and loads player coins with full error handling using pcall for every DataStore operation.",
            "code": """-- DataStore coin save/load system with full error handling
local DataStoreService = game:GetService("DataStoreService")
local Players = game:GetService("Players")

local coinStore = DataStoreService:GetDataStore("PlayerCoins")

-- Load player data
local function loadCoins(player)
\tlocal success, coins = pcall(function()
\t\treturn coinStore:GetAsync("Player_" .. player.UserId)
\tend)

\tif success then
\t\tlocal leaderstats = Instance.new("Folder")
\t\tleaderstats.Name = "leaderstats"
\t\tleaderstats.Parent = player

\t\tlocal coinValue = Instance.new("IntValue")
\t\tcoinValue.Name = "Coins"
\t\tcoinValue.Value = coins or 0
\t\tcoinValue.Parent = leaderstats
\telse
\t\twarn("Failed to load coins for " .. player.Name .. ": " .. tostring(coins))
\tend
end

-- Save player data
local function saveCoins(player)
\tlocal leaderstats = player:FindFirstChild("leaderstats")
\tif not leaderstats then return end

\tlocal coinValue = leaderstats:FindFirstChild("Coins")
\tif not coinValue then return end

\tlocal success, err = pcall(function()
\t\tcoinStore:SetAsync("Player_" .. player.UserId, coinValue.Value)
\tend)

\tif not success then
\t\twarn("Failed to save coins for " .. player.Name .. ": " .. tostring(err))
\tend
end

-- Connect events
Players.PlayerAdded:Connect(loadCoins)

Players.PlayerRemoving:Connect(saveCoins)

-- Auto-save every 5 minutes
task.spawn(function()
\twhile true do
\t\ttask.wait(300)
\t\tfor _, player in Players:GetPlayers() do
\t\t\tsaveCoins(player)
\t\tend
\tend
end)
""",
        },
        {
            "task": "Write a Roblox NPC that follows the nearest player using PathfindingService, with proper nil checks and connection cleanup.",
            "code": """-- NPC pathfinding system that follows nearest player
local PathfindingService = game:GetService("PathfindingService")
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")

local npc = script.Parent
local humanoid = npc:WaitForChild("Humanoid")
local rootPart = npc:WaitForChild("HumanoidRootPart")

-- Configuration
local FOLLOW_RANGE = 50
local UPDATE_INTERVAL = 1
local PATHFINDING_PARAMS = {
\tAgentRadius = 2,
\tAgentHeight = 5,
\tAgentCanJump = true,
}

-- Find the nearest player within range
local function findNearestPlayer()
\tlocal nearestPlayer = nil
\tlocal nearestDistance = FOLLOW_RANGE

\tfor _, player in Players:GetPlayers() do
\t\tlocal character = player.Character
\t\tif not character then continue end

\t\tlocal playerRoot = character:FindFirstChild("HumanoidRootPart")
\t\tif not playerRoot then continue end

\t\tlocal distance = (playerRoot.Position - rootPart.Position).Magnitude
\t\tif distance < nearestDistance then
\t\t\tnearestDistance = distance
\t\t\tnearestPlayer = player
\t\tend
\tend

\treturn nearestPlayer
end

-- Move NPC along path to target
local function moveToTarget(targetPosition)
\tlocal path = PathfindingService:CreatePath(PATHFINDING_PARAMS)

\tlocal success, err = pcall(function()
\t\tpath:ComputeAsync(rootPart.Position, targetPosition)
\tend)

\tif not success then
\t\twarn("Pathfinding failed: " .. tostring(err))
\t\treturn
\tend

\tif path.Status ~= Enum.PathStatus.Success then
\t\treturn
\tend

\tlocal waypoints = path:GetWaypoints()
\tfor _, waypoint in waypoints do
\t\thumanoid:MoveTo(waypoint.Position)
\t\tif waypoint.Action == Enum.PathWaypointAction.Jump then
\t\t\thumanoid.Jump = true
\t\tend
\t\thumanoid.MoveToFinished:Wait()
\tend
end

-- Main follow loop
task.spawn(function()
\twhile true do
\t\ttask.wait(UPDATE_INTERVAL)

\t\tlocal target = findNearestPlayer()
\t\tif target and target.Character then
\t\t\tlocal targetRoot = target.Character:FindFirstChild("HumanoidRootPart")
\t\t\tif targetRoot then
\t\t\t\tmoveToTarget(targetRoot.Position)
\t\t\tend
\t\tend
\tend
end)
""",
        },
        {
            "task": "Write a Roblox inventory UI system with a toggle button, grid layout, and item interaction using modern Roblox API patterns.",
            "code": """-- Inventory UI system with toggle, grid layout, and item interaction
local Players = game:GetService("Players")
local UserInputService = game:GetService("UserInputService")
local TweenService = game:GetService("TweenService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local player = Players.LocalPlayer
local playerGui = player:WaitForChild("PlayerGui")

-- Configuration
local GRID_COLUMNS = 5
local SLOT_SIZE = UDim2.new(0, 60, 0, 60)
local SLOT_PADDING = 5

-- Create UI
local function createInventoryUI()
\tlocal screenGui = Instance.new("ScreenGui")
\tscreenGui.Name = "InventoryGui"
\tscreenGui.ResetOnSpawn = false

\tlocal frame = Instance.new("Frame")
\tframe.Name = "InventoryFrame"
\tframe.Size = UDim2.new(0, 350, 0, 400)
\tframe.Position = UDim2.new(0.5, -175, 0.5, -200)
\tframe.BackgroundColor3 = Color3.fromRGB(30, 30, 30)
\tframe.BackgroundTransparency = 0.1
\tframe.Visible = false
\tframe.Parent = screenGui

\tlocal corner = Instance.new("UICorner")
\tcorner.CornerRadius = UDim.new(0, 8)
\tcorner.Parent = frame

\t-- Title
\tlocal title = Instance.new("TextLabel")
\ttitle.Name = "Title"
\ttitle.Size = UDim2.new(1, 0, 0, 40)
\ttitle.BackgroundTransparency = 1
\ttitle.Text = "Inventory"
\ttitle.TextColor3 = Color3.fromRGB(255, 255, 255)
\ttitle.TextSize = 20
\ttitle.Font = Enum.Font.GothamBold
\ttitle.Parent = frame

\t-- Scrolling grid
\tlocal scrollFrame = Instance.new("ScrollingFrame")
\tscrollFrame.Name = "Grid"
\tscrollFrame.Size = UDim2.new(1, -20, 1, -50)
\tscrollFrame.Position = UDim2.new(0, 10, 0, 45)
\tscrollFrame.BackgroundTransparency = 1
\tscrollFrame.ScrollBarThickness = 4
\tscrollFrame.Parent = frame

\tlocal gridLayout = Instance.new("UIGridLayout")
\tgridLayout.CellSize = SLOT_SIZE
\tgridLayout.CellPadding = UDim2.new(0, SLOT_PADDING, 0, SLOT_PADDING)
\tgridLayout.SortOrder = Enum.SortOrder.LayoutOrder
\tgridLayout.Parent = scrollFrame

\t-- Toggle button
\tlocal toggleBtn = Instance.new("TextButton")
\ttoggleBtn.Name = "ToggleInventory"
\ttoggleBtn.Size = UDim2.new(0, 100, 0, 35)
\ttoggleBtn.Position = UDim2.new(0.5, -50, 1, -45)
\ttoggleBtn.BackgroundColor3 = Color3.fromRGB(50, 50, 50)
\ttoggleBtn.Text = "Inventory"
\ttoggleBtn.TextColor3 = Color3.fromRGB(255, 255, 255)
\ttoggleBtn.TextSize = 14
\ttoggleBtn.Font = Enum.Font.Gotham
\ttoggleBtn.Parent = screenGui

\tlocal btnCorner = Instance.new("UICorner")
\tbtnCorner.CornerRadius = UDim.new(0, 6)
\tbtnCorner.Parent = toggleBtn

\tscreenGui.Parent = playerGui

\treturn screenGui, frame, scrollFrame, toggleBtn
end

local screenGui, frame, scrollFrame, toggleBtn = createInventoryUI()

-- Toggle visibility with tween
local function toggleInventory()
\tlocal isVisible = frame.Visible
\tif isVisible then
\t\tlocal tween = TweenService:Create(frame, TweenInfo.new(0.2), {
\t\t\tBackgroundTransparency = 1,
\t\t})
\t\ttween:Play()
\t\ttween.Completed:Wait()
\t\tframe.Visible = false
\telse
\t\tframe.Visible = true
\t\tframe.BackgroundTransparency = 1
\t\tlocal tween = TweenService:Create(frame, TweenInfo.new(0.2), {
\t\t\tBackgroundTransparency = 0.1,
\t\t})
\t\ttween:Play()
\tend
end

-- Add item slot to grid
local function addItemSlot(itemName, itemIcon, quantity)
\tlocal slot = Instance.new("TextButton")
\tslot.Name = itemName
\tslot.BackgroundColor3 = Color3.fromRGB(60, 60, 60)
\tslot.Text = ""

\tlocal slotCorner = Instance.new("UICorner")
\tslotCorner.CornerRadius = UDim.new(0, 4)
\tslotCorner.Parent = slot

\tlocal icon = Instance.new("ImageLabel")
\ticon.Size = UDim2.new(0.8, 0, 0.8, 0)
\ticon.Position = UDim2.new(0.1, 0, 0.05, 0)
\ticon.BackgroundTransparency = 1
\ticon.Image = itemIcon or ""
\ticon.Parent = slot

\tlocal countLabel = Instance.new("TextLabel")
\tcountLabel.Size = UDim2.new(0.5, 0, 0.3, 0)
\tcountLabel.Position = UDim2.new(0.5, 0, 0.7, 0)
\tcountLabel.BackgroundTransparency = 1
\tcountLabel.Text = tostring(quantity or 1)
\tcountLabel.TextColor3 = Color3.fromRGB(255, 255, 255)
\tcountLabel.TextSize = 12
\tcountLabel.Font = Enum.Font.GothamBold
\tcountLabel.Parent = slot

\t-- Hover effect
\tslot.MouseEnter:Connect(function()
\t\tTweenService:Create(slot, TweenInfo.new(0.1), {
\t\t\tBackgroundColor3 = Color3.fromRGB(80, 80, 80),
\t\t}):Play()
\tend)

\tslot.MouseLeave:Connect(function()
\t\tTweenService:Create(slot, TweenInfo.new(0.1), {
\t\t\tBackgroundColor3 = Color3.fromRGB(60, 60, 60),
\t\t}):Play()
\tend)

\t-- Click handler
\tslot.Activated:Connect(function()
\t\tprint("Selected item: " .. itemName)
\tend)

\tslot.Parent = scrollFrame
end

-- Connect toggle
toggleBtn.Activated:Connect(toggleInventory)

-- Keyboard shortcut (Tab to toggle)
UserInputService.InputBegan:Connect(function(input, processed)
\tif processed then return end
\tif input.KeyCode == Enum.KeyCode.Tab then
\t\ttoggleInventory()
\tend
end)

-- Example: populate with items
addItemSlot("Sword", "rbxassetid://123456", 1)
addItemSlot("Shield", "rbxassetid://123457", 1)
addItemSlot("Potion", "rbxassetid://123458", 5)
addItemSlot("Gold", "rbxassetid://123459", 100)
""",
        },
        {
            "task": "Write a Roblox round-based game system with a lobby, countdown timer, teleporting players, and round management.",
            "code": """-- Round-based game system with lobby, timer, teleport, and round management
local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local ServerStorage = game:GetService("ServerStorage")

-- Configuration
local LOBBY_WAIT = 15
local MIN_PLAYERS = 2
local ROUND_LENGTH = 120
local INTERMISSION = 10

-- Map spawns
local lobbySpawn = workspace:WaitForChild("LobbySpawn")
local arenaSpawns = workspace:WaitForChild("ArenaSpawns"):GetChildren()

-- Status display
local statusValue = Instance.new("StringValue")
statusValue.Name = "GameStatus"
statusValue.Parent = ReplicatedStorage

local roundActive = false

-- Teleport player to position
local function teleportPlayer(player, position)
\tlocal character = player.Character
\tif not character then return end

\tlocal rootPart = character:FindFirstChild("HumanoidRootPart")
\tif not rootPart then return end

\trootPart.CFrame = CFrame.new(position)
end

-- Teleport all players to arena
local function teleportToArena()
\tlocal players = Players:GetPlayers()
\tfor i, player in players do
\t\tlocal spawnIndex = ((i - 1) % #arenaSpawns) + 1
\t\tlocal spawn = arenaSpawns[spawnIndex]
\t\tteleportPlayer(player, spawn.Position + Vector3.new(0, 5, 0))
\tend
end

-- Teleport all players to lobby
local function teleportToLobby()
\tfor _, player in Players:GetPlayers() do
\t\tteleportPlayer(player, lobbySpawn.Position + Vector3.new(0, 5, 0))
\tend
end

-- Get alive players in round
local function getAlivePlayers()
\tlocal alive = {}
\tfor _, player in Players:GetPlayers() do
\t\tlocal character = player.Character
\t\tif not character then continue end

\t\tlocal humanoid = character:FindFirstChild("Humanoid")
\t\tif humanoid and humanoid.Health > 0 then
\t\t\ttable.insert(alive, player)
\t\tend
\tend
\treturn alive
end

-- Main game loop
task.spawn(function()
\twhile true do
\t\t-- Lobby phase: wait for minimum players
\t\tstatusValue.Value = "Waiting for players (" .. #Players:GetPlayers() .. "/" .. MIN_PLAYERS .. ")"

\t\twhile #Players:GetPlayers() < MIN_PLAYERS do
\t\t\tstatusValue.Value = "Waiting for players (" .. #Players:GetPlayers() .. "/" .. MIN_PLAYERS .. ")"
\t\t\ttask.wait(1)
\t\tend

\t\t-- Countdown
\t\tfor countdown = LOBBY_WAIT, 1, -1 do
\t\t\tstatusValue.Value = "Round starts in " .. countdown .. "s"
\t\t\ttask.wait(1)

\t\t\tif #Players:GetPlayers() < MIN_PLAYERS then
\t\t\t\tbreak
\t\t\tend
\t\tend

\t\t-- Check if still enough players
\t\tif #Players:GetPlayers() < MIN_PLAYERS then
\t\t\tcontinue
\t\tend

\t\t-- Start round
\t\troundActive = true
\t\tstatusValue.Value = "Round starting!"
\t\ttask.wait(2)

\t\tteleportToArena()

\t\t-- Round timer
\t\tlocal timeLeft = ROUND_LENGTH
\t\twhile timeLeft > 0 and roundActive do
\t\t\tlocal alive = getAlivePlayers()
\t\t\tstatusValue.Value = "Round: " .. timeLeft .. "s | Alive: " .. #alive

\t\t\tif #alive <= 1 then
\t\t\t\troundActive = false
\t\t\t\tif #alive == 1 then
\t\t\t\t\tstatusValue.Value = alive[1].Name .. " wins!"
\t\t\t\telse
\t\t\t\t\tstatusValue.Value = "No winner!"
\t\t\t\tend
\t\t\t\tbreak
\t\t\tend

\t\t\ttask.wait(1)
\t\t\ttimeLeft = timeLeft - 1
\t\tend

\t\t-- Round ended
\t\troundActive = false
\t\tif timeLeft <= 0 then
\t\t\tstatusValue.Value = "Time's up!"
\t\tend

\t\ttask.wait(3)
\t\tteleportToLobby()

\t\t-- Intermission
\t\tfor countdown = INTERMISSION, 1, -1 do
\t\t\tstatusValue.Value = "Intermission: " .. countdown .. "s"
\t\t\ttask.wait(1)
\t\tend
\tend
end)
""",
        },
        {
            "task": "Write a Roblox pet follow system where pets orbit around the player using CFrame math and smooth tweening.",
            "code": """-- Pet follow system with orbital movement using CFrame math
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local TweenService = game:GetService("TweenService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

-- Configuration
local ORBIT_RADIUS = 5
local ORBIT_SPEED = 1.5
local FLOAT_HEIGHT = 2
local BOB_AMPLITUDE = 0.3
local BOB_SPEED = 2
local LERP_SPEED = 0.1

-- Create a pet model
local function createPet(petName)
\tlocal pet = Instance.new("Part")
\tpet.Name = petName
\tpet.Size = Vector3.new(2, 2, 2)
\tpet.Shape = Enum.PartType.Ball
\tpet.Material = Enum.Material.Neon
\tpet.Color = Color3.fromRGB(255, 200, 50)
\tpet.Anchored = true
\tpet.CanCollide = false
\tpet.CastShadow = false
\tpet.Parent = workspace

\tlocal billboard = Instance.new("BillboardGui")
\tbillboard.Size = UDim2.new(0, 100, 0, 30)
\tbillboard.StudsOffset = Vector3.new(0, 2, 0)
\tbillboard.Parent = pet

\tlocal nameLabel = Instance.new("TextLabel")
\tnameLabel.Size = UDim2.new(1, 0, 1, 0)
\tnameLabel.BackgroundTransparency = 1
\tnameLabel.Text = petName
\tnameLabel.TextColor3 = Color3.fromRGB(255, 255, 255)
\tnameLabel.TextStrokeTransparency = 0.5
\tnameLabel.TextSize = 14
\tnameLabel.Font = Enum.Font.GothamBold
\tnameLabel.Parent = billboard

\treturn pet
end

-- Pet controller class
local PetController = {}
PetController.__index = PetController

function PetController.new(player, petName, orbitIndex)
\tlocal self = setmetatable({}, PetController)
\tself.player = player
\tself.pet = createPet(petName)
\tself.orbitIndex = orbitIndex or 0
\tself.angle = (orbitIndex or 0) * (math.pi * 2 / 3)
\tself.targetCFrame = CFrame.new()
\tself.connection = nil
\treturn self
end

function PetController:start()
\tself.connection = RunService.Heartbeat:Connect(function(dt)
\t\tself:update(dt)
\tend)
end

function PetController:update(dt)
\tlocal character = self.player.Character
\tif not character then return end

\tlocal rootPart = character:FindFirstChild("HumanoidRootPart")
\tif not rootPart then return end

\t-- Calculate orbital position
\tself.angle = self.angle + (ORBIT_SPEED * dt)
\tlocal bobOffset = math.sin(tick() * BOB_SPEED) * BOB_AMPLITUDE

\tlocal orbitX = math.cos(self.angle) * ORBIT_RADIUS
\tlocal orbitZ = math.sin(self.angle) * ORBIT_RADIUS
\tlocal orbitY = FLOAT_HEIGHT + bobOffset

\tlocal targetPos = rootPart.Position + Vector3.new(orbitX, orbitY, orbitZ)

\t-- Smooth lerp to target
\tself.targetCFrame = CFrame.new(targetPos)
\tlocal currentCFrame = self.pet.CFrame
\tself.pet.CFrame = currentCFrame:Lerp(self.targetCFrame, LERP_SPEED)

\t-- Face the player
\tlocal lookAt = CFrame.lookAt(self.pet.Position, rootPart.Position)
\tself.pet.CFrame = CFrame.new(self.pet.Position) * lookAt.Rotation
end

function PetController:destroy()
\tif self.connection then
\t\tself.connection:Disconnect()
\t\tself.connection = nil
\tend
\tif self.pet then
\t\tself.pet:Destroy()
\tend
end

-- Manage pets for all players
local playerPets = {}

local function onPlayerAdded(player)
\tplayer.CharacterAdded:Connect(function()
\t\ttask.wait(1)

\t\t-- Give each player 2 pets
\t\tlocal pets = {}
\t\tfor i = 1, 2 do
\t\t\tlocal petName = player.Name .. "'s Pet " .. i
\t\t\tlocal controller = PetController.new(player, petName, i)
\t\t\tcontroller:start()
\t\t\ttable.insert(pets, controller)
\t\tend

\t\tplayerPets[player] = pets
\tend)
end

local function onPlayerRemoving(player)
\tlocal pets = playerPets[player]
\tif pets then
\t\tfor _, controller in pets do
\t\t\tcontroller:destroy()
\t\tend
\t\tplayerPets[player] = nil
\tend
end

-- Connect events
Players.PlayerAdded:Connect(onPlayerAdded)
Players.PlayerRemoving:Connect(onPlayerRemoving)

-- Handle existing players
for _, player in Players:GetPlayers() do
\tonPlayerAdded(player)
end
""",
        },
    ]

    samples = []
    for item in synthetic_tasks[:n_samples]:
        samples.append(format_sample(item["task"], item["code"]))

    print(f"  Generated {len(samples)} synthetic examples")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Format Luau files into training JSONL")
    parser.add_argument("--no-synthetic", action="store_true", help="Skip synthetic data generation")
    parser.add_argument("--use-llm", action="store_true", help="Use Mistral for task descriptions")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max real samples to process")
    args = parser.parse_args()

    print("FORMATTING LUAU TRAINING DATA")
    print(f"Output: {OUTPUT_FILE}\n")

    all_samples = []

    # Process real files
    print("Phase 1: Processing real Luau files...")
    real_samples = process_raw_files(use_llm=args.use_llm, max_samples=args.max_samples)
    all_samples.extend(real_samples)
    print(f"  Real samples: {len(real_samples)}")

    # Generate synthetic hardening
    if not args.no_synthetic:
        print("\nPhase 2: Generating synthetic hardening examples...")
        synthetic = generate_synthetic_hardening(n_samples=50)
        all_samples.extend(synthetic)
        print(f"  Synthetic samples: {len(synthetic)}")

    # Write JSONL
    print(f"\nWriting {len(all_samples)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(all_samples)} training samples written.")
    print(f"  Real: {len(real_samples)}")
    if not args.no_synthetic:
        print(f"  Synthetic: {len(all_samples) - len(real_samples)}")
    print(f"  File: {OUTPUT_FILE}")
    print(f"  Size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
