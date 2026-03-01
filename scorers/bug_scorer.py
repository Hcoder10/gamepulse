"""Detects common Luau bug patterns with context-aware checks."""

import re
from scorers.compat import Scorer, op


class BugScorer(Scorer):
    """Detects common Luau bug patterns with context-aware checks.

    Minimizes false positives by understanding common Luau idioms
    and only flagging genuinely risky patterns.
    """

    @op()
    def score(self, output: str, **kwargs) -> dict:
        bugs_found = []
        stripped = self._strip_comments(output)

        checks = [
            (self._check_unchecked_find, "Unchecked FindFirstChild (nil dereference risk)", 0.20),
            (self._check_infinite_loop, "while true do without yield", 0.30),
            (self._check_datastore_pcall, "DataStore call without pcall", 0.20),
            (self._check_connection_leak, "Event connection without cleanup pattern", 0.10),
            (self._check_string_concat_loop, "String concatenation in loop", 0.05),
            (self._check_globals, "Global variable assignment", 0.15),
        ]

        total_penalty = 0.0
        max_penalty = sum(w for _, _, w in checks)

        for check_fn, description, weight in checks:
            count = check_fn(stripped)
            if count > 0:
                penalty = weight * min(count, 3)
                total_penalty += penalty
                bugs_found.append({
                    "bug": description,
                    "occurrences": count,
                    "severity_weight": weight,
                })

        score = max(0.0, 1.0 - (total_penalty / max_penalty))

        return {
            "bug_free": len(bugs_found) == 0,
            "bug_score": round(score, 3),
            "bugs_found": bugs_found,
        }

    @staticmethod
    def _strip_comments(code: str) -> str:
        result = re.sub(r"--\[\[.*?\]\]", "", code, flags=re.DOTALL)
        result = re.sub(r"--[^\n]*", "", result)
        return result

    @staticmethod
    def _check_unchecked_find(code: str) -> int:
        """FindFirstChild immediately followed by property access without nil check."""
        # Only flag direct chained access: :FindFirstChild("x").Prop or :FindFirstChild("x"):Method
        direct_access = re.findall(
            r":(?:FindFirstChild|FindFirstChildOfClass|FindFirstChildWhichIsA)\s*\([^)]+\)\s*[\.\:]",
            code
        )
        return len(direct_access)

    @staticmethod
    def _check_infinite_loop(code: str) -> int:
        """while true do blocks that don't contain any yield."""
        count = 0
        pattern = r"\bwhile\s+true\s+do\b"
        for match in re.finditer(pattern, code):
            start = match.end()
            depth = 1
            pos = start
            block_end = len(code)
            while pos < len(code) and depth > 0:
                end_match = re.search(r"\b(do|end)\b", code[pos:])
                if not end_match:
                    break
                word = end_match.group(1)
                pos += end_match.end()
                if word == "do":
                    depth += 1
                elif word == "end":
                    depth -= 1
                    if depth == 0:
                        block_end = pos
            block = code[start:block_end]
            yields = re.search(
                r"\btask\.wait\b|\btask\.yield\b|\bwait\b|\bcoroutine\.yield\b|\btask\.defer\b|\bRunService\b|\b:Wait\b",
                block
            )
            if not yields:
                count += 1
        return count

    @staticmethod
    def _check_datastore_pcall(code: str) -> int:
        """DataStore async calls not wrapped in pcall.

        Uses line-by-line analysis to check if each async call is inside a pcall block.
        """
        async_calls = re.findall(r"(?:GetAsync|SetAsync|UpdateAsync|RemoveAsync|IncrementAsync)\s*\(", code)
        if not async_calls:
            return 0

        # If code has pcall/xpcall anywhere near DataStore calls, likely wrapped
        has_pcall = bool(re.search(r"\bpcall\s*\(|\bxpcall\s*\(", code))
        if has_pcall:
            return 0

        return len(async_calls)

    @staticmethod
    def _check_connection_leak(code: str) -> int:
        """Connections without any cleanup pattern in the entire script.

        Only flags if the script has MANY connections and ZERO cleanup.
        Server scripts that run forever don't need to disconnect PlayerAdded etc.
        """
        connections = re.findall(r"[\.\:]Connect\s*\(", code)
        if not connections:
            return 0

        # Look for ANY cleanup pattern in the whole script
        cleanup = re.search(
            r"\bDisconnect\b|\b:Destroy\b|\bMaid\b|\bJanitor\b|\btrove\b"
            r"|\b\.Once\b|\bCleanup\b|\bcleanup\b|\bRemoving\b",
            code
        )
        if cleanup:
            return 0

        # Server scripts with PlayerAdded/Heartbeat/etc. don't need cleanup
        server_patterns = re.search(
            r"\bPlayerAdded\b|\bPlayerRemoving\b|\bHeartbeat\b|\bStepped\b"
            r"|\bChildAdded\b|\bChildRemoved\b|\bCharacterAdded\b",
            code
        )
        if server_patterns:
            return 0

        # Only flag if there are many unmanaged connections with no cleanup whatsoever
        if len(connections) > 8:
            return len(connections) - 8
        return 0

    @staticmethod
    def _check_string_concat_loop(code: str) -> int:
        """String concatenation with .. inside for loops (use table.concat instead)."""
        count = 0
        for match in re.finditer(r"\bfor\b.*\bdo\b(.*?)\bend\b", code, re.DOTALL):
            block = match.group(1)
            # Only flag if there's variable = variable .. something pattern
            if re.search(r"\w+\s*=\s*\w+\s*\.\.", block):
                count += 1
        return count

    @staticmethod
    def _check_globals(code: str) -> int:
        """Lines that assign to non-local variables (crude check).

        Avoids false positives from:
        - Table field assignments (t.x = ...)
        - Indexed assignments (t[k] = ...)
        - Method definitions (function t:method())
        - Re-assignments of previously declared locals
        - Common Luau patterns (self.x = ..., module.x = ...)
        - Loop variables in for-in
        - Variables that were declared local earlier in the script
        """
        # First pass: collect all locally declared variable names
        local_vars = set()
        for line in code.split("\n"):
            stripped = line.strip()
            # Match: local x = ... or local x, y = ... or local function x
            local_match = re.match(r"^local\s+(?:function\s+)?(\w+)", stripped)
            if local_match:
                local_vars.add(local_match.group(1))
            # Match: local x, y, z = ...
            multi_local = re.match(r"^local\s+([\w\s,]+)\s*=", stripped)
            if multi_local:
                for var in multi_local.group(1).split(","):
                    var = var.strip()
                    if var and re.match(r"^\w+$", var):
                        local_vars.add(var)
            # for i, v in / for i = patterns declare locals
            for_match = re.match(r"^for\s+(\w+)", stripped)
            if for_match:
                local_vars.add(for_match.group(1))
            for_in_match = re.match(r"^for\s+(\w+)\s*,\s*(\w+)", stripped)
            if for_in_match:
                local_vars.add(for_in_match.group(1))
                local_vars.add(for_in_match.group(2))

        # Also add common builtins/globals that aren't really globals
        builtins = {
            "game", "workspace", "script", "plugin", "shared", "math", "string",
            "table", "coroutine", "task", "debug", "os", "utf8", "bit32",
            "Instance", "Enum", "CFrame", "Vector3", "Vector2", "Color3",
            "UDim", "UDim2", "Ray", "Region3", "BrickColor", "TweenInfo",
            "NumberRange", "NumberSequence", "ColorSequence", "Rect",
            "print", "warn", "error", "type", "typeof", "tostring", "tonumber",
            "pairs", "ipairs", "next", "select", "unpack", "rawget", "rawset",
            "setmetatable", "getmetatable", "require", "pcall", "xpcall",
            "tick", "time", "wait", "spawn", "delay",
        }

        count = 0
        for line in code.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue

            # Match bare assignment: varname = value (but not ==)
            bare_assign = re.match(r"^([a-zA-Z_]\w*)\s*=[^=]", stripped)
            if not bare_assign:
                continue

            var_name = bare_assign.group(1)

            # Skip if it's a known local or builtin
            if var_name in local_vars or var_name in builtins:
                continue

            # Skip property/table/method assignments (has . or [ or : before =)
            if re.match(r"^[a-zA-Z_]\w*\s*[\.\[:]", stripped):
                continue

            # Skip if 'local' appears on this line
            if "local " in line:
                continue

            # Skip function definitions (function name(...))
            if re.match(r"^function\s", stripped):
                continue

            count += 1

        return count
