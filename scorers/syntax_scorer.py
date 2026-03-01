import re
from scorers.compat import Scorer, op


class SyntaxScorer(Scorer):
    """Validates Luau syntax with context-aware parsing.

    Strips comments and string literals before checking structure,
    preventing false positives from content inside strings/comments.
    """

    @op()
    def score(self, output: str, **kwargs) -> dict:
        issues = []
        stripped = self._strip_comments_and_strings(output)

        # Bracket matching (on stripped code)
        bracket_issues = self._check_brackets(stripped)
        issues.extend(bracket_issues)

        # Block keyword balance
        block_issues = self._check_block_balance(stripped)
        issues.extend(block_issues)

        # Python-ism detection (on original to catch them in real code)
        python_issues = self._check_python_isms(stripped)
        issues.extend(python_issues)

        # Severity-weighted scoring instead of flat deduction
        severity = {
            "bracket": 0.15,
            "block": 0.12,
            "python": 0.08,
        }
        total_penalty = 0.0
        for issue in issues:
            for key, weight in severity.items():
                if key in issue.lower():
                    total_penalty += weight
                    break
            else:
                total_penalty += 0.05

        score = max(0.0, 1.0 - total_penalty)

        return {
            "syntax_valid": score >= 0.85,
            "syntax_score": round(score, 3),
            "syntax_issues": issues,
        }

    @staticmethod
    def _strip_comments_and_strings(code: str) -> str:
        """Remove all comments and string literals for accurate structural analysis."""
        # Multi-line strings [[...]]
        result = re.sub(r"\[\[.*?\]\]", '""', code, flags=re.DOTALL)
        # Multi-line comments --[[...]]
        result = re.sub(r"--\[\[.*?\]\]", "", result, flags=re.DOTALL)
        # Single-line comments
        result = re.sub(r"--[^\n]*", "", result)
        # Double-quoted strings
        result = re.sub(r'"(?:[^"\\]|\\.)*"', '""', result)
        # Single-quoted strings
        result = re.sub(r"'(?:[^'\\]|\\.)*'", "''", result)
        # Backtick strings (Luau interpolation)
        result = re.sub(r"`(?:[^`\\]|\\.)*`", "``", result)
        return result

    @staticmethod
    def _check_brackets(code: str) -> list[str]:
        issues = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        stack = []
        for i, ch in enumerate(code):
            if ch in pairs:
                stack.append((ch, i))
            elif ch in pairs.values():
                if not stack:
                    issues.append(f"Bracket: unmatched closing '{ch}'")
                    break  # One issue is enough signal
                else:
                    opener, _ = stack.pop()
                    if pairs[opener] != ch:
                        issues.append(f"Bracket: '{opener}' closed with '{ch}'")
                        break
        if len(stack) > 3:
            issues.append(f"Bracket: {len(stack)} unclosed brackets")
        elif len(stack) > 0:
            for opener, pos in stack[-2:]:  # Report at most 2
                issues.append(f"Bracket: unclosed '{opener}'")
        return issues

    @staticmethod
    def _check_block_balance(code: str) -> list[str]:
        issues = []

        # Count block openers that need 'end'
        # function, if, for X do, while X do each need one 'end'
        # Standalone 'do...end' blocks also need 'end'
        # 'do' after for/while is part of that statement, not a separate block
        functions = len(re.findall(r"\bfunction\b", code))
        ifs = len(re.findall(r"\bif\b", code))
        elseifs = len(re.findall(r"\belseif\b", code))
        ifs -= elseifs  # elseif contains 'if' but doesn't open a new block
        fors = len(re.findall(r"\bfor\b", code))
        whiles = len(re.findall(r"\bwhile\b", code))
        # Standalone do blocks (do not preceded by for/while on the same logical line)
        all_dos = len(re.findall(r"\bdo\b", code))
        attached_dos = fors + whiles  # do that belongs to for/while
        standalone_dos = max(0, all_dos - attached_dos)

        block_openers = functions + ifs + fors + whiles + standalone_dos

        repeats = len(re.findall(r"\brepeat\b", code))
        untils = len(re.findall(r"\buntil\b", code))
        ends = len(re.findall(r"\bend\b", code))

        expected_ends = block_openers
        diff = abs(ends - expected_ends)

        # Allow small tolerance for complex scripts (1 off in 200+ line scripts)
        if diff > 2:
            issues.append(f"Block: {block_openers} openers but {ends} 'end' (off by {diff})")
        elif diff > 0:
            # Minor imbalance - lower severity
            issues.append(f"Block: minor imbalance ({block_openers} openers, {ends} ends)")

        if abs(repeats - untils) > 0:
            issues.append(f"Block: {repeats} repeat vs {untils} until")

        return issues

    @staticmethod
    def _check_python_isms(code: str) -> list[str]:
        issues = []
        patterns = [
            (r"\bdef\s+\w+\s*\(", "Python: 'def' keyword"),
            (r"\bclass\s+\w+\s*[:\(]", "Python: 'class' keyword"),
            (r"^import\s+\w+", "Python: 'import' statement"),
            (r"\bTrue\b", "Python: 'True' (use 'true')"),
            (r"\bFalse\b", "Python: 'False' (use 'false')"),
            (r"\bNone\b", "Python: 'None' (use 'nil')"),
            (r"[^~]=\s*!=\s*", "Python: '!=' (use '~=')"),
            (r"\belif\b", "Python: 'elif' (use 'elseif')"),
        ]
        for pattern, msg in patterns:
            if re.search(pattern, code, re.MULTILINE):
                issues.append(msg)
        return issues
