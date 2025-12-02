"""Core pattern composition interfaces."""

from .combinator import PatternCombinator
from .composite_pattern import CompositePattern
from .pattern import Pattern

__all__ = ["Pattern", "CompositePattern", "PatternCombinator"]
