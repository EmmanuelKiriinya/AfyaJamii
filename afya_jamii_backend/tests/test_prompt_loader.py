"""Tests for prompt loading.

Prompts live in JSON files outside the source, so the loader is the only thing
standing between a typo in a template and a malformed request to the model.
"""

import json

import pytest

from app.prompt_loader import DEFAULT_PROMPT_DIR, PromptError, PromptLibrary


@pytest.fixture
def library(tmp_path):
    return PromptLibrary(tmp_path)


def write_prompt(directory, name, document):
    path = directory / f"{name}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# ── The bundled prompt ─────────────────────────────────────────────────────

def test_bundled_prompt_loads_and_renders():
    prompt = PromptLibrary(DEFAULT_PROMPT_DIR).get("clinical_assistant")

    assert prompt.id == "clinical_assistant"
    assert prompt.input_variables == ("context", "history", "question")

    rendered = prompt.render(
        {"context": "BP 170/110", "history": "", "question": "What should I do?"}
    )
    assert "BP 170/110" in rendered
    assert "What should I do?" in rendered
    # The included contacts file must have been substituted in.
    assert "1199" in rendered
    # And no placeholder may survive rendering.
    assert "{emergency_contacts}" not in rendered


# ── Template handling ──────────────────────────────────────────────────────

def test_template_lines_are_joined(tmp_path, library):
    write_prompt(
        tmp_path,
        "joined",
        {"input_variables": ["q"], "template": ["first", "second {q}"]},
    )
    assert library.get("joined").template == "first\nsecond {q}"


def test_template_may_be_a_plain_string(tmp_path, library):
    write_prompt(tmp_path, "plain", {"input_variables": ["q"], "template": "ask {q}"})
    assert library.get("plain").render({"q": "why"}) == "ask why"


def test_included_file_is_substituted(tmp_path, library):
    (tmp_path / "extra.md").write_text("Call 999", encoding="utf-8")
    write_prompt(
        tmp_path,
        "with_include",
        {
            "input_variables": ["q"],
            "includes": {"extra": "extra.md"},
            "template": "{q}\n{extra}",
        },
    )
    assert "Call 999" in library.get("with_include").render({"q": "help"})


def test_braces_in_included_prose_survive_rendering(tmp_path, library):
    """Included Markdown may contain braces that are not placeholders."""
    (tmp_path / "extra.md").write_text("Use {this} literally", encoding="utf-8")
    write_prompt(
        tmp_path,
        "braces",
        {
            "input_variables": ["q"],
            "includes": {"extra": "extra.md"},
            "template": "{q} {extra}",
        },
    )
    assert "{this}" in library.get("braces").render({"q": "ok"})


# ── Validation ─────────────────────────────────────────────────────────────

def test_undeclared_placeholder_is_rejected(tmp_path, library):
    write_prompt(
        tmp_path,
        "mismatch",
        {"input_variables": ["q"], "template": "{q} and {surprise}"},
    )
    with pytest.raises(PromptError, match="surprise"):
        library.get("mismatch")


def test_missing_input_variables_is_rejected(tmp_path, library):
    write_prompt(tmp_path, "bare", {"template": "no variables declared"})
    with pytest.raises(PromptError, match="input_variables"):
        library.get("bare")


def test_invalid_json_is_reported_clearly(tmp_path, library):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(PromptError, match="not valid JSON"):
        library.get("broken")


def test_missing_prompt_lists_what_is_available(tmp_path, library):
    write_prompt(tmp_path, "present", {"input_variables": ["q"], "template": "{q}"})
    with pytest.raises(PromptError, match="present"):
        library.get("absent")


def test_missing_include_is_reported(tmp_path, library):
    write_prompt(
        tmp_path,
        "dangling",
        {
            "input_variables": ["q"],
            "includes": {"extra": "nowhere.md"},
            "template": "{q} {extra}",
        },
    )
    with pytest.raises(PromptError, match="nowhere.md"):
        library.get("dangling")


def test_rendering_without_a_value_raises(tmp_path, library):
    write_prompt(tmp_path, "needs", {"input_variables": ["a", "b"], "template": "{a}{b}"})
    with pytest.raises(PromptError, match="b"):
        library.get("needs").render({"a": "only"})


@pytest.mark.parametrize("name", ["../escape", "sub/dir", ".hidden"])
def test_path_traversal_is_refused(library, name):
    with pytest.raises(PromptError, match="Invalid prompt name"):
        library.get(name)


# ── Caching ────────────────────────────────────────────────────────────────

def test_prompts_are_cached_between_calls(tmp_path, library):
    write_prompt(tmp_path, "cached", {"input_variables": ["q"], "template": "{q}"})
    assert library.get("cached") is library.get("cached")


def test_reload_on_change_picks_up_edits(tmp_path):
    library = PromptLibrary(tmp_path, reload_on_change=True)
    path = write_prompt(tmp_path, "live", {"input_variables": ["q"], "template": "one {q}"})
    assert library.get("live").template == "one {q}"

    import os
    import time

    write_prompt(tmp_path, "live", {"input_variables": ["q"], "template": "two {q}"})
    # Ensure the mtime actually differs on filesystems with coarse resolution.
    os.utime(path, (time.time() + 1, time.time() + 1))

    assert library.get("live").template == "two {q}"
