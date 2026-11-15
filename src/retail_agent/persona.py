"""Persona: how the assistant communicates (tone, style, report template), editable by non-developers.

`config/persona.yaml` is re-read on every turn, so a change applies from the next message without a
restart or redeploy (requirement 8). If an edit breaks the file, the last valid persona stays in use
and the error is kept for reporting, so a typo cannot take the assistant down.

The persona changes how the assistant talks, never what it may access: access and privacy rules are
enforced in code and stated in the fixed part of the system prompt, which says the persona cannot
override them.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from retail_agent.policy import ConfigFileError, load_yaml_model

PERSONA_FILE = "persona.yaml"


class Persona(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    updated_by: str
    tone: str = Field(min_length=1)
    style: tuple[str, ...] = ()
    report_template: str = Field(min_length=1)


class PersonaStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._last_good: Persona | None = None
        self.last_error: str | None = None  # set while the file on disk is invalid

    def get(self) -> Persona:
        """The persona on disk now, or the last valid one if the file is currently broken."""
        try:
            self._last_good = load_yaml_model(Persona, self.path)
            self.last_error = None
        except ConfigFileError as exc:
            self.last_error = str(exc)
            if self._last_good is None:
                raise
        return self._last_good
