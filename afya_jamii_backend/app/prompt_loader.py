"""Loading and validation of prompt templates stored outside the codebase.

A prompt is a JSON document with the following shape::

    {
      "id": "clinical_assistant",
      "version": "2.0.0",
      "input_variables": ["context", "history", "question"],
      "includes": {"emergency_contacts": "emergency_contacts.md"},
      "template": ["line one", "line two", ...]
    }

``template`` may be a single string or a list of lines that are joined with
newlines; the list form keeps diffs readable when the prompt is edited.

Entries in ``includes`` map a placeholder name to a sibling file. The file's
contents are substituted into the template once, at load time, so that large
reference material (emergency contact directories, formularies) can be
maintained as ordinary Markdown.

Templates are cached after their first successful load. Set
``PROMPT_RELOAD_ON_CHANGE=true`` to re-read a template whenever the file's
modification time changes, which is convenient while iterating on wording.
"""

from __future__ import annotations

import json
import logging
import string
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


class PromptError(RuntimeError):
    """Raised when a prompt file is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class Prompt:
    """A validated prompt template ready to be formatted."""

    id: str
    version: str
    template: str
    input_variables: tuple[str, ...]
    source: Path

    def render(self, values: Mapping[str, Any]) -> str:
        """Substitute ``values`` into the template.

        Uses :class:`string.Template` semantics via ``str.format`` on a
        pre-validated variable set, so a missing variable fails loudly here
        rather than silently reaching the model.
        """
        missing = [name for name in self.input_variables if name not in values]
        if missing:
            raise PromptError(
                f"Prompt '{self.id}' is missing values for: {', '.join(missing)}"
            )
        return self.template.format(**{k: values[k] for k in self.input_variables})

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Prompt {self.id} v{self.version} from {self.source.name}>"


class PromptLibrary:
    """Thread-safe, caching reader for the prompt directory."""

    def __init__(self, directory: Path | str | None = None, *, reload_on_change: bool = False):
        self.directory = Path(directory) if directory else DEFAULT_PROMPT_DIR
        self.reload_on_change = reload_on_change
        self._cache: dict[str, tuple[float, Prompt]] = {}
        self._lock = Lock()

    def get(self, name: str) -> Prompt:
        """Return the prompt named ``name`` (without the ``.json`` suffix)."""
        path = self._resolve(name)
        mtime = path.stat().st_mtime

        with self._lock:
            cached = self._cache.get(name)
            if cached is not None:
                cached_mtime, prompt = cached
                if not self.reload_on_change or cached_mtime == mtime:
                    return prompt

            prompt = self._load(name, path)
            self._cache[name] = (mtime, prompt)

        logger.info("Loaded prompt '%s' version %s from %s", prompt.id, prompt.version, path)
        return prompt

    def _resolve(self, name: str) -> Path:
        # Reject path traversal: prompts are addressed by bare name only.
        if "/" in name or "\\" in name or name.startswith("."):
            raise PromptError(f"Invalid prompt name: {name!r}")

        path = self.directory / f"{name}.json"
        if not path.is_file():
            raise PromptError(
                f"Prompt '{name}' not found at {path}. "
                f"Available prompts: {', '.join(self.available()) or 'none'}"
            )
        return path

    def available(self) -> list[str]:
        if not self.directory.is_dir():
            return []
        return sorted(p.stem for p in self.directory.glob("*.json"))

    def _load(self, name: str, path: Path) -> Prompt:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PromptError(f"Prompt '{name}' is not valid JSON: {exc}") from exc

        raw_template = document.get("template")
        if isinstance(raw_template, list):
            template = "\n".join(str(line) for line in raw_template)
        elif isinstance(raw_template, str):
            template = raw_template
        else:
            raise PromptError(
                f"Prompt '{name}' must define 'template' as a string or list of strings"
            )

        template = self._apply_includes(name, path, template, document.get("includes") or {})

        declared = tuple(document.get("input_variables") or ())
        if not declared:
            raise PromptError(f"Prompt '{name}' must declare 'input_variables'")

        self._check_placeholders(name, template, declared)

        return Prompt(
            id=str(document.get("id") or name),
            version=str(document.get("version") or "0.0.0"),
            template=template,
            input_variables=declared,
            source=path,
        )

    def _apply_includes(
        self, name: str, path: Path, template: str, includes: Mapping[str, str]
    ) -> str:
        for placeholder, filename in includes.items():
            if "/" in filename or "\\" in filename or filename.startswith("."):
                raise PromptError(f"Prompt '{name}' has an invalid include path: {filename!r}")

            include_path = path.parent / filename
            if not include_path.is_file():
                raise PromptError(
                    f"Prompt '{name}' includes '{filename}', which does not exist at {include_path}"
                )

            content = include_path.read_text(encoding="utf-8").strip()
            token = "{" + placeholder + "}"
            if token not in template:
                logger.warning(
                    "Prompt '%s' declares include '%s' but never references %s",
                    name,
                    filename,
                    token,
                )
            # Escape braces in included prose so str.format leaves them alone.
            template = template.replace(token, content.replace("{", "{{").replace("}", "}}"))
        return template

    @staticmethod
    def _check_placeholders(name: str, template: str, declared: tuple[str, ...]) -> None:
        """Fail fast if the template and its declared variables disagree."""
        found = {
            field
            for _, field, _, _ in string.Formatter().parse(template)
            if field
        }
        undeclared = found - set(declared)
        if undeclared:
            raise PromptError(
                f"Prompt '{name}' uses undeclared placeholders: {', '.join(sorted(undeclared))}"
            )

        unused = set(declared) - found
        if unused:
            logger.warning(
                "Prompt '%s' declares unused input variables: %s",
                name,
                ", ".join(sorted(unused)),
            )
