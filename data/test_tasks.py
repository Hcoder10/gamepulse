"""15 Roblox coding tasks across 6 categories for evaluation."""

TEST_TASKS = [
    # --- NPC Behavior (2) ---
    {
        "task_description": "Create an NPC that patrols between 4 waypoints using PathfindingService. "
        "The NPC should walk to each waypoint in order, wait 2 seconds at each one, "
        "then loop back to the first waypoint. Handle blocked paths by recomputing.",
        "category": "npc_behavior",
    },
    {
        "task_description": "Create a shop NPC with a ProximityPrompt. When a player interacts, "
        "open a shop GUI showing 3 items with prices. The NPC should face the player during interaction. "
        "Use RemoteEvents for purchase validation on the server.",
        "category": "npc_behavior",
    },
    # --- Game Mechanics (3) ---
    {
        "task_description": "Implement a round-based game system. Players wait in a lobby, "
        "then teleport to an arena when enough players join (minimum 2). "
        "The round lasts 120 seconds. Last player standing wins. "
        "Track kills and display a leaderboard using leaderstats.",
        "category": "game_mechanics",
    },
    {
        "task_description": "Create a coin collection system. Spawn 20 coins randomly across the map. "
        "When a player touches a coin, award 1 point to their leaderstats and destroy the coin. "
        "Respawn coins every 30 seconds. Play a sound effect on collection.",
        "category": "game_mechanics",
    },
    {
        "task_description": "Build a crafting system where players combine 2 items from their inventory "
        "to create a new item. Use a ModuleScript for recipes. "
        "Validate crafting on the server via RemoteFunction. "
        "Update the player's inventory after successful crafting.",
        "category": "game_mechanics",
    },
    # --- UI Scripts (2) ---
    {
        "task_description": "Create an animated health bar UI that smoothly tweens when the player takes damage. "
        "Show current HP / max HP as text. Change the bar color from green to yellow to red "
        "based on health percentage. Add a damage flash effect.",
        "category": "ui",
    },
    {
        "task_description": "Build a settings menu GUI with toggles for music, sound effects, and shadows. "
        "Save player preferences using DataStoreService so settings persist between sessions. "
        "Apply settings immediately when toggled.",
        "category": "ui",
    },
    # --- Physics / CFrame (2) ---
    {
        "task_description": "Create a door that smoothly opens when a player steps on a pressure plate "
        "and closes when they step off. Use TweenService to animate the door's CFrame. "
        "The door should rotate 90 degrees on its hinge.",
        "category": "physics_cframe",
    },
    {
        "task_description": "Implement a cannon that shoots a projectile in the direction it faces. "
        "Use CFrame.lookAt for aiming and apply velocity using VectorForce or BodyVelocity. "
        "Add a particle effect trail to the projectile. Destroy the projectile after 5 seconds.",
        "category": "physics_cframe",
    },
    # --- Data Persistence (2) ---
    {
        "task_description": "Create a player data save system using DataStoreService. "
        "Save player level, experience, and coins when they leave. "
        "Load data when they join. Handle edge cases: new players, failed saves, "
        "and BindToClose for server shutdown.",
        "category": "data_persistence",
    },
    {
        "task_description": "Implement a global leaderboard using OrderedDataStore. "
        "Display the top 10 players by score on an in-game SurfaceGui. "
        "Update the leaderboard every 60 seconds. Handle DataStore request limits.",
        "category": "data_persistence",
    },
    # --- Remote Events (2) ---
    {
        "task_description": "Create a chat command system using RemoteEvents. "
        "Players type /heal, /speed, or /tp in chat. The client sends the command "
        "to the server via RemoteEvent. The server validates and executes the command. "
        "Add cooldowns to prevent spam.",
        "category": "remote_events",
    },
    {
        "task_description": "Build a trading system between two players. Player A requests a trade, "
        "Player B accepts or declines via a GUI. Both players select items, "
        "then confirm the trade. Use RemoteEvents for all client-server communication. "
        "Validate that both players still have their items before swapping.",
        "category": "remote_events",
    },
    # --- Mixed / Advanced (2) ---
    {
        "task_description": "Create a pet follow system. Players can equip a pet that follows them "
        "using BodyPosition or AlignPosition. The pet bobs up and down with a sine wave animation. "
        "Store the equipped pet in DataStoreService. Support multiple pet types.",
        "category": "advanced",
    },
    {
        "task_description": "Implement a procedural terrain generator that creates a flat island "
        "with random trees and rocks using math.noise for height variation. "
        "Generate terrain in chunks as players explore. Optimize by only generating "
        "chunks within render distance.",
        "category": "advanced",
    },
]
