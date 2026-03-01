import re
from scorers.compat import Scorer, op


class QualityScorer(Scorer):
    """Evaluates code quality with calibrated, realistic scoring."""

    @op()
    def score(self, output: str, **kwargs) -> dict:
        details = {}
        lines = output.split("\n")
        non_empty = [l for l in lines if l.strip()]
        total_lines = len(non_empty)

        # 1. Comment presence (realistic: any meaningful comments = good)
        comment_lines = sum(1 for l in lines if re.match(r"\s*--", l))
        density = comment_lines / max(total_lines, 1)
        if comment_lines == 0:
            comment_score = 0.2
        elif comment_lines < 3:
            comment_score = 0.5
        elif density < 0.05:
            comment_score = 0.6
        elif density <= 0.35:
            comment_score = 1.0
        else:
            comment_score = 0.8
        details["comment_lines"] = comment_lines
        details["comment_density"] = round(density, 3)
        details["comment_score"] = round(comment_score, 3)

        # 2. Code organization (header comments, section separators, or grouped declarations)
        # Broader detection: any comment that labels a section
        section_markers = re.findall(
            r"--\s*[=\-]{3,}|"           # -- ====  or  -- ----
            r"--\s*\[\[|"                 # --[[ multi-line
            r"--\s*[A-Z][A-Za-z\s]{2,}:", # -- Services:, -- Constants:
            output,
        )
        has_top_comment = bool(re.match(r"\s*--", lines[0])) if lines else False
        has_service_block = bool(re.search(r"(local\s+\w+\s*=\s*game:GetService.*\n){2,}", output))

        org_score = 0.3  # baseline
        if has_top_comment:
            org_score += 0.25
        if len(section_markers) >= 1:
            org_score += 0.2
        if has_service_block:
            org_score += 0.25
        org_score = min(org_score, 1.0)
        details["organization_score"] = round(org_score, 3)

        # 3. Naming conventions
        service_refs = re.findall(r"local\s+(\w+)\s*=\s*game:GetService", output)
        pascal_ok = sum(1 for s in service_refs if s[0].isupper()) if service_refs else 1
        pascal_total = max(len(service_refs), 1)

        local_vars = re.findall(r"local\s+(\w+)\s*=(?!\s*game:GetService)", output)
        camel_ok = sum(1 for v in local_vars if v[0].islower() or v.startswith("_")) if local_vars else 1
        camel_total = max(len(local_vars), 1)

        naming_score = (pascal_ok / pascal_total * 0.5) + (camel_ok / camel_total * 0.5)
        details["naming_score"] = round(naming_score, 3)

        # 4. Error handling
        has_pcall = bool(re.search(r"\bpcall\b|\bxpcall\b", output))
        needs_error_handling = bool(re.search(
            r"DataStore|HttpService|RemoteFunction|:InvokeServer|:InvokeClient|"
            r"GetAsync|SetAsync|UpdateAsync|TeleportService",
            output
        ))
        if needs_error_handling:
            error_score = 1.0 if has_pcall else 0.15
        else:
            error_score = 1.0 if has_pcall else 0.75  # bonus for using pcall even when not required
        details["error_handling_score"] = round(error_score, 3)

        # 5. Code completeness (not too short, not absurdly long)
        if total_lines < 8:
            length_score = 0.3
        elif total_lines < 15:
            length_score = 0.6
        elif total_lines <= 300:
            length_score = 1.0
        else:
            length_score = 0.85
        details["total_lines"] = len(lines)
        details["length_score"] = round(length_score, 3)

        # Weighted composite
        composite = (
            comment_score * 0.20
            + org_score * 0.15
            + naming_score * 0.25
            + error_score * 0.20
            + length_score * 0.20
        )

        return {
            "quality_score": round(composite, 3),
            "quality_details": details,
        }
