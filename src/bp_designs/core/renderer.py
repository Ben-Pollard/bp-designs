"""Rendering context for structured output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import svgwrite


class RenderStyle(BaseModel):
    """Base class for structured rendering parameters using Pydantic."""

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> RenderStyle:
        """Create a style object from a dictionary."""
        return cls(**params)

    def get_svg_attributes(self) -> dict[str, Any]:
        """Extract parameters that are not explicitly defined as fields,
        treating them as raw SVG attributes.
        """
        # Get all fields defined in the model (including inherited ones)
        known_fields = self.model_fields.keys()
        # Return everything else from the model's __dict__ (extra fields)
        return {k: v for k, v in self.__dict__.items() if k not in known_fields}


class RenderingContext:
    """Wraps a drawing backend to provide structured rendering (groups, etc.)."""

    def __init__(self, dwg: svgwrite.Drawing):
        self.dwg = dwg
        self.stack = [dwg]

    def push_group(self, name: str, **kwargs) -> Any:
        """Create and enter a new SVG group."""
        group = self.dwg.g(id=name, **kwargs)
        self.stack[-1].add(group)
        self.stack.append(group)
        return group

    def pop_group(self):
        """Exit the current SVG group."""
        if len(self.stack) > 1:
            self.stack.pop()

    def add(self, element):
        """Add an element to the current group/drawing."""
        self.stack[-1].add(element)

    @property
    def current_target(self):
        """Return the current group or drawing."""
        return self.stack[-1]
