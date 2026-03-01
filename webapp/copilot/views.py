"""
Views for Luau Copilot.

Page views render templates; API views handle AJAX requests and return JSON.
"""
import json
import traceback

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from copilot import services


# ---------------------------------------------------------------------------
# Page views
# ---------------------------------------------------------------------------

def generate_page(request):
    """Interactive code generation page."""
    test_tasks = services.get_test_tasks()
    example_tasks = [t["task_description"][:80] for t in test_tasks[:6]]
    return render(request, "copilot/generate.html", {
        "example_tasks": example_tasks,
        "test_tasks": test_tasks,
    })


def about_page(request):
    """About page explaining the pipeline."""
    return render(request, "copilot/about.html")


def plugin_page(request):
    """Studio plugin download page with install instructions."""
    from pathlib import Path
    plugin_path = Path(__file__).resolve().parent.parent.parent / "studio_plugin" / "LuauCopilot.lua"
    plugin_code = ""
    if plugin_path.exists():
        plugin_code = plugin_path.read_text(encoding="utf-8")

    # Determine server URL for the plugin
    server_url = request.build_absolute_uri("/").rstrip("/")

    return render(request, "copilot/plugin.html", {
        "plugin_code": plugin_code,
        "server_url": server_url,
    })


def plugin_download(request):
    """Serve the plugin .lua file as a download."""
    from pathlib import Path
    plugin_path = Path(__file__).resolve().parent.parent.parent / "studio_plugin" / "LuauCopilot.lua"
    if plugin_path.exists():
        content = plugin_path.read_text(encoding="utf-8")
        response = HttpResponse(content, content_type="application/octet-stream")
        response["Content-Disposition"] = 'attachment; filename="LuauCopilot.lua"'
        return response
    return HttpResponse("Plugin file not found", status=404)


# ---------------------------------------------------------------------------
# AJAX API views
# ---------------------------------------------------------------------------

@csrf_exempt
@require_POST
def api_generate(request):
    """Generate code from a task description using the chosen model."""
    try:
        body = json.loads(request.body)
        task = body.get("task", "").strip()
        model = body.get("model", "rft").strip()

        if not task:
            return JsonResponse({"error": "Task description is required."}, status=400)

        if model not in ("sft", "rft"):
            return JsonResponse({"error": "Invalid model choice. Use 'sft' or 'rft'."}, status=400)

        code = services.generate_code(task, model)
        scores = services.score_code(code)

        return JsonResponse({
            "code": code,
            "scores": scores["scores"],
            "avg_score": scores["avg_score"],
            "issues": scores["issues"],
            "passed": scores["passed"],
        })

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": f"Generation failed: {str(e)}"}, status=500)


@csrf_exempt
@require_POST
def api_score(request):
    """Score a piece of code with all four scorers."""
    try:
        body = json.loads(request.body)
        code = body.get("code", "").strip()

        if not code:
            return JsonResponse({"error": "Code is required."}, status=400)

        scores = services.score_code(code)

        return JsonResponse({
            "scores": scores["scores"],
            "avg_score": scores["avg_score"],
            "issues": scores["issues"],
            "passed": scores["passed"],
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": f"Scoring failed: {str(e)}"}, status=500)


@csrf_exempt
@require_POST
def api_self_correct(request):
    """Run the self-correction loop on a piece of code."""
    try:
        body = json.loads(request.body)
        task = body.get("task", "").strip()
        code = body.get("code", "").strip()

        if not task or not code:
            return JsonResponse(
                {"error": "Both task description and code are required."},
                status=400,
            )

        result = services.self_correct(task, code, max_rounds=3)

        return JsonResponse({
            "original_code": result["original_code"],
            "final_code": result["final_code"],
            "rounds": result["rounds"],
            "original_scores": result["original_scores"],
            "original_avg": result["original_avg"],
            "final_scores": result["final_scores"],
            "final_avg": result["final_avg"],
            "improved": result["improved"],
            "passed": result["passed"],
            "history": result["history"],
        })

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse(
            {"error": f"Self-correction failed: {str(e)}"},
            status=500,
        )


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------

def chat_page(request):
    """Agentic chat interface — main experience."""
    return render(request, "copilot/chat.html")


def pipeline_page(request):
    """Self-improvement pipeline dashboard."""
    iteration_log = services.load_iteration_log()
    forge_log = services.load_forge_log()
    prompt_versions = services.load_prompt_versions()
    pipeline_status = services.get_pipeline_status()
    return render(request, "copilot/pipeline.html", {
        "iteration_log": json.dumps(iteration_log),
        "forge_log": json.dumps(forge_log),
        "prompt_versions": json.dumps(prompt_versions),
        "pipeline_status": pipeline_status,
    })


# ---------------------------------------------------------------------------
# Agentic API
# ---------------------------------------------------------------------------

@csrf_exempt
@require_POST
def api_agentic_generate(request):
    """Full agentic pipeline: generate -> score -> self-correct -> queue for Studio."""
    try:
        body = json.loads(request.body)
        task = body.get("task", "").strip()
        model = body.get("model", "rft").strip()
        auto_correct = body.get("auto_correct", True)
        session_id = body.get("session_id", None)

        if not task:
            return JsonResponse({"error": "Task description is required."}, status=400)

        result = services.agentic_generate(task, model, auto_correct, session_id)
        return JsonResponse(result)

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": f"Generation failed: {str(e)}"}, status=500)


# ---------------------------------------------------------------------------
# Pipeline API
# ---------------------------------------------------------------------------

def api_pipeline_status(request):
    """Return pipeline dashboard status data."""
    try:
        status = services.get_pipeline_status()
        return JsonResponse(status)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


def api_pipeline_prompts(request):
    """Return all prompt versions."""
    try:
        versions = services.load_prompt_versions()
        return JsonResponse({"versions": versions})
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


def api_pipeline_prompt_detail(request, version):
    """Return a specific prompt version's content."""
    content = services.get_prompt_version(version)
    if content is None:
        return JsonResponse({"error": f"Version {version} not found"}, status=404)
    return JsonResponse({"version": version, "content": content})


# ---------------------------------------------------------------------------
# Studio Plugin API
# ---------------------------------------------------------------------------

@csrf_exempt
def api_studio_poll(request):
    """Studio plugin polls for pending commands."""
    session_id = request.GET.get("session_id", "")
    if not session_id:
        return JsonResponse({"error": "session_id required"}, status=400)
    commands = services.poll_studio_commands(session_id)
    return JsonResponse({"commands": commands})


@csrf_exempt
@require_POST
def api_studio_report(request):
    """Studio plugin reports command results."""
    try:
        body = json.loads(request.body)
        command_id = body.get("command_id", "")
        success = body.get("success", False)
        message = body.get("message", "")
        services.report_studio_result(command_id, success, message)
        return JsonResponse({"ok": True})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def api_studio_send(request):
    """Send code directly to Studio via the command queue."""
    try:
        body = json.loads(request.body)
        session_id = body.get("session_id", "")
        code = body.get("code", "")
        script_name = body.get("name", "GeneratedScript")
        parent = body.get("parent", "ServerScriptService")
        script_class = body.get("class", "Script")

        if not session_id:
            return JsonResponse({"error": "session_id required"}, status=400)
        if not code:
            return JsonResponse({"error": "code required"}, status=400)

        cmd_id = services.queue_studio_command(session_id, "insert_script", {
            "name": script_name,
            "source": code,
            "parent": parent,
            "class": script_class,
        })
        return JsonResponse({"ok": True, "command_id": cmd_id})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_studio_heartbeat(request):
    """Studio plugin heartbeat — confirms connection."""
    session_id = request.GET.get("session_id", "")
    last_poll = services.get_studio_last_poll(session_id)
    return JsonResponse({"connected": True, "session_id": session_id, "last_poll": last_poll})
