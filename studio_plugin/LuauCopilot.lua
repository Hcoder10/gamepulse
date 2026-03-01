--[[
    Luau Copilot - Roblox Studio Plugin

    Connects to the Luau Copilot web app and automatically inserts
    AI-generated scripts into your game.

    SETUP:
    1. Enable HttpService in Game Settings > Security
    2. Copy this file into your Plugins folder
    3. Enter the server URL and Session ID from the web app
    4. Click "Connect"

    The plugin will poll the server for commands and execute them:
    - insert_script: Creates a new Script/LocalScript/ModuleScript
    - create_instance: Creates any Instance type
    - modify_property: Changes a property on an existing instance
]]

-- Configuration
local POLL_INTERVAL = 2 -- seconds between polls
local DEFAULT_SERVER = "https://web-production-aac10.up.railway.app"

-- Services
local HttpService = game:GetService("HttpService")
local ServerScriptService = game:GetService("ServerScriptService")
local ServerStorage = game:GetService("ServerStorage")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local StarterGui = game:GetService("StarterGui")
local StarterPlayer = game:GetService("StarterPlayer")
local Selection = game:GetService("Selection")
local ChangeHistoryService = game:GetService("ChangeHistoryService")

-- Plugin setup
local toolbar = plugin:CreateToolbar("Luau Copilot")
local toggleButton = toolbar:CreateButton(
    "Luau Copilot",
    "Connect to Luau Copilot AI",
    "rbxassetid://4458901886" -- code icon
)

-- State
local connected = false
local polling = false
local serverUrl = DEFAULT_SERVER
local sessionId = ""

-- Parent resolution
local PARENT_MAP = {
    ServerScriptService = ServerScriptService,
    ServerStorage = ServerStorage,
    ReplicatedStorage = ReplicatedStorage,
    StarterGui = StarterGui,
    StarterPlayerScripts = StarterPlayer:WaitForChild("StarterPlayerScripts"),
    StarterCharacterScripts = StarterPlayer:WaitForChild("StarterCharacterScripts"),
    Workspace = workspace,
}

-- UI
local widget = plugin:CreateDockWidgetPluginGui(
    "LuauCopilot",
    DockWidgetPluginGuiInfo.new(
        Enum.InitialDockState.Right,
        false, -- initially disabled
        false, -- override previous state
        300,   -- default width
        400,   -- default height
        250,   -- min width
        200    -- min height
    )
)
widget.Title = "Luau Copilot"

-- Build UI
local frame = Instance.new("Frame")
frame.Size = UDim2.new(1, 0, 1, 0)
frame.BackgroundColor3 = Color3.fromRGB(17, 24, 39)
frame.BorderSizePixel = 0
frame.Parent = widget

local layout = Instance.new("UIListLayout")
layout.Padding = UDim.new(0, 8)
layout.SortOrder = Enum.SortOrder.LayoutOrder
layout.Parent = frame

local padding = Instance.new("UIPadding")
padding.PadLeft = UDim.new(0, 12)
padding.PadRight = UDim.new(0, 12)
padding.PadTop = UDim.new(0, 12)
padding.PadBottom = UDim.new(0, 12)
padding.Parent = frame

-- Title
local title = Instance.new("TextLabel")
title.Size = UDim2.new(1, 0, 0, 28)
title.BackgroundTransparency = 1
title.Text = "Luau Copilot"
title.TextColor3 = Color3.fromRGB(0, 162, 255)
title.TextSize = 18
title.Font = Enum.Font.GothamBold
title.TextXAlignment = Enum.TextXAlignment.Left
title.LayoutOrder = 1
title.Parent = frame

-- Server URL input
local serverLabel = Instance.new("TextLabel")
serverLabel.Size = UDim2.new(1, 0, 0, 16)
serverLabel.BackgroundTransparency = 1
serverLabel.Text = "Server URL"
serverLabel.TextColor3 = Color3.fromRGB(156, 163, 175)
serverLabel.TextSize = 12
serverLabel.Font = Enum.Font.Gotham
serverLabel.TextXAlignment = Enum.TextXAlignment.Left
serverLabel.LayoutOrder = 2
serverLabel.Parent = frame

local serverInput = Instance.new("TextBox")
serverInput.Size = UDim2.new(1, 0, 0, 32)
serverInput.BackgroundColor3 = Color3.fromRGB(30, 37, 54)
serverInput.BorderSizePixel = 0
serverInput.Text = DEFAULT_SERVER
serverInput.TextColor3 = Color3.fromRGB(209, 213, 219)
serverInput.PlaceholderText = "http://localhost:8000"
serverInput.PlaceholderColor3 = Color3.fromRGB(107, 114, 128)
serverInput.TextSize = 13
serverInput.Font = Enum.Font.Code
serverInput.ClearTextOnFocus = false
serverInput.LayoutOrder = 3
serverInput.Parent = frame

local serverCorner = Instance.new("UICorner")
serverCorner.CornerRadius = UDim.new(0, 6)
serverCorner.Parent = serverInput

local serverPadding = Instance.new("UIPadding")
serverPadding.PadLeft = UDim.new(0, 8)
serverPadding.PadRight = UDim.new(0, 8)
serverPadding.Parent = serverInput

-- Session ID input
local sessionLabel = Instance.new("TextLabel")
sessionLabel.Size = UDim2.new(1, 0, 0, 16)
sessionLabel.BackgroundTransparency = 1
sessionLabel.Text = "Session ID (from web app)"
sessionLabel.TextColor3 = Color3.fromRGB(156, 163, 175)
sessionLabel.TextSize = 12
sessionLabel.Font = Enum.Font.Gotham
sessionLabel.TextXAlignment = Enum.TextXAlignment.Left
sessionLabel.LayoutOrder = 4
sessionLabel.Parent = frame

local sessionInput = Instance.new("TextBox")
sessionInput.Size = UDim2.new(1, 0, 0, 32)
sessionInput.BackgroundColor3 = Color3.fromRGB(30, 37, 54)
sessionInput.BorderSizePixel = 0
sessionInput.Text = ""
sessionInput.TextColor3 = Color3.fromRGB(209, 213, 219)
sessionInput.PlaceholderText = "Paste session ID here"
sessionInput.PlaceholderColor3 = Color3.fromRGB(107, 114, 128)
sessionInput.TextSize = 13
sessionInput.Font = Enum.Font.Code
sessionInput.ClearTextOnFocus = false
sessionInput.LayoutOrder = 5
sessionInput.Parent = frame

local sessionCorner = Instance.new("UICorner")
sessionCorner.CornerRadius = UDim.new(0, 6)
sessionCorner.Parent = sessionInput

local sessionPadding = Instance.new("UIPadding")
sessionPadding.PadLeft = UDim.new(0, 8)
sessionPadding.PadRight = UDim.new(0, 8)
sessionPadding.Parent = sessionInput

-- Connect button
local connectBtn = Instance.new("TextButton")
connectBtn.Size = UDim2.new(1, 0, 0, 36)
connectBtn.BackgroundColor3 = Color3.fromRGB(0, 162, 255)
connectBtn.BorderSizePixel = 0
connectBtn.Text = "Connect"
connectBtn.TextColor3 = Color3.fromRGB(255, 255, 255)
connectBtn.TextSize = 14
connectBtn.Font = Enum.Font.GothamBold
connectBtn.LayoutOrder = 6
connectBtn.Parent = frame

local connectCorner = Instance.new("UICorner")
connectCorner.CornerRadius = UDim.new(0, 8)
connectCorner.Parent = connectBtn

-- Status label
local statusLabel = Instance.new("TextLabel")
statusLabel.Size = UDim2.new(1, 0, 0, 20)
statusLabel.BackgroundTransparency = 1
statusLabel.Text = "Status: Disconnected"
statusLabel.TextColor3 = Color3.fromRGB(107, 114, 128)
statusLabel.TextSize = 12
statusLabel.Font = Enum.Font.Gotham
statusLabel.TextXAlignment = Enum.TextXAlignment.Left
statusLabel.LayoutOrder = 7
statusLabel.Parent = frame

-- Log area
local logFrame = Instance.new("ScrollingFrame")
logFrame.Size = UDim2.new(1, 0, 1, -220)
logFrame.BackgroundColor3 = Color3.fromRGB(11, 15, 25)
logFrame.BorderSizePixel = 0
logFrame.ScrollBarThickness = 4
logFrame.ScrollBarImageColor3 = Color3.fromRGB(51, 65, 85)
logFrame.CanvasSize = UDim2.new(0, 0, 0, 0)
logFrame.AutomaticCanvasSize = Enum.AutomaticSize.Y
logFrame.LayoutOrder = 8
logFrame.Parent = frame

local logCorner = Instance.new("UICorner")
logCorner.CornerRadius = UDim.new(0, 6)
logCorner.Parent = logFrame

local logLayout = Instance.new("UIListLayout")
logLayout.Padding = UDim.new(0, 2)
logLayout.SortOrder = Enum.SortOrder.LayoutOrder
logLayout.Parent = logFrame

local logPadding = Instance.new("UIPadding")
logPadding.PadLeft = UDim.new(0, 8)
logPadding.PadRight = UDim.new(0, 8)
logPadding.PadTop = UDim.new(0, 6)
logPadding.PadBottom = UDim.new(0, 6)
logPadding.Parent = logFrame

local logCount = 0

local function addLog(text, color)
    logCount = logCount + 1
    local label = Instance.new("TextLabel")
    label.Size = UDim2.new(1, 0, 0, 0)
    label.AutomaticSize = Enum.AutomaticSize.Y
    label.BackgroundTransparency = 1
    label.Text = os.date("%H:%M:%S") .. " " .. text
    label.TextColor3 = color or Color3.fromRGB(156, 163, 175)
    label.TextSize = 11
    label.Font = Enum.Font.Code
    label.TextXAlignment = Enum.TextXAlignment.Left
    label.TextWrapped = true
    label.LayoutOrder = logCount
    label.Parent = logFrame

    -- Auto-scroll
    task.defer(function()
        logFrame.CanvasPosition = Vector2.new(0, logFrame.AbsoluteCanvasSize.Y)
    end)
end

-- Command handlers
local function handleInsertScript(payload)
    local scriptName = payload.name or "GeneratedScript"
    local source = payload.source or ""
    local parentName = payload.parent or "ServerScriptService"
    local className = payload["class"] or "Script"

    local parent = PARENT_MAP[parentName] or ServerScriptService

    ChangeHistoryService:SetWaypoint("LuauCopilot: Insert " .. scriptName)

    local scriptObj = Instance.new(className)
    scriptObj.Name = scriptName
    scriptObj.Source = source
    scriptObj.Parent = parent

    Selection:Set({scriptObj})

    addLog("Created " .. className .. " '" .. scriptName .. "' in " .. parentName, Color3.fromRGB(16, 185, 129))
    return true, "Script created successfully"
end

local function handleCreateInstance(payload)
    local className = payload["class"] or "Part"
    local instanceName = payload.name or className
    local parentName = payload.parent or "Workspace"
    local properties = payload.properties or {}

    local parent = PARENT_MAP[parentName] or workspace

    ChangeHistoryService:SetWaypoint("LuauCopilot: Create " .. instanceName)

    local obj = Instance.new(className)
    obj.Name = instanceName

    for prop, val in pairs(properties) do
        pcall(function()
            obj[prop] = val
        end)
    end

    obj.Parent = parent
    Selection:Set({obj})

    addLog("Created " .. className .. " '" .. instanceName .. "'", Color3.fromRGB(16, 185, 129))
    return true, "Instance created"
end

local COMMAND_HANDLERS = {
    insert_script = handleInsertScript,
    create_instance = handleCreateInstance,
}

-- Networking
local function reportResult(commandId, success, message)
    pcall(function()
        HttpService:PostAsync(
            serverUrl .. "/copilot/api/studio/report/",
            HttpService:JSONEncode({
                command_id = commandId,
                success = success,
                message = message,
            }),
            Enum.HttpContentType.ApplicationJson
        )
    end)
end

local function pollForCommands()
    local ok, result = pcall(function()
        local response = HttpService:GetAsync(
            serverUrl .. "/copilot/api/studio/poll/?session_id=" .. sessionId
        )
        return HttpService:JSONDecode(response)
    end)

    if not ok then
        return
    end

    local commands = result.commands or {}
    for _, cmd in ipairs(commands) do
        local handler = COMMAND_HANDLERS[cmd.type]
        if handler then
            local success, message = handler(cmd.payload)
            reportResult(cmd.id, success, message)
        else
            addLog("Unknown command: " .. tostring(cmd.type), Color3.fromRGB(226, 35, 26))
            reportResult(cmd.id, false, "Unknown command type")
        end
    end
end

local function startPolling()
    polling = true
    task.spawn(function()
        while polling do
            pollForCommands()
            task.wait(POLL_INTERVAL)
        end
    end)
end

local function stopPolling()
    polling = false
end

-- Connect/disconnect
local function connect()
    serverUrl = serverInput.Text:gsub("/$", "") -- strip trailing slash
    sessionId = sessionInput.Text

    if sessionId == "" then
        addLog("Enter a session ID from the web app", Color3.fromRGB(226, 35, 26))
        return
    end

    -- Test connection
    local ok, result = pcall(function()
        local response = HttpService:GetAsync(
            serverUrl .. "/copilot/api/studio/heartbeat/?session_id=" .. sessionId
        )
        return HttpService:JSONDecode(response)
    end)

    if ok and result.connected then
        connected = true
        connectBtn.Text = "Disconnect"
        connectBtn.BackgroundColor3 = Color3.fromRGB(226, 35, 26)
        statusLabel.Text = "Status: Connected"
        statusLabel.TextColor3 = Color3.fromRGB(16, 185, 129)
        addLog("Connected to " .. serverUrl, Color3.fromRGB(16, 185, 129))
        startPolling()
    else
        addLog("Failed to connect: " .. tostring(result), Color3.fromRGB(226, 35, 26))
    end
end

local function disconnect()
    connected = false
    stopPolling()
    connectBtn.Text = "Connect"
    connectBtn.BackgroundColor3 = Color3.fromRGB(0, 162, 255)
    statusLabel.Text = "Status: Disconnected"
    statusLabel.TextColor3 = Color3.fromRGB(107, 114, 128)
    addLog("Disconnected", Color3.fromRGB(156, 163, 175))
end

connectBtn.MouseButton1Click:Connect(function()
    if connected then
        disconnect()
    else
        connect()
    end
end)

-- Toggle widget with toolbar button
toggleButton.Click:Connect(function()
    widget.Enabled = not widget.Enabled
end)

-- Cleanup on unload
plugin.Unloading:Connect(function()
    stopPolling()
end)

addLog("Luau Copilot plugin loaded", Color3.fromRGB(0, 162, 255))
addLog("Enter server URL and session ID to connect", Color3.fromRGB(156, 163, 175))
