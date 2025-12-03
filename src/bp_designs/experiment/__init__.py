"""Experimentation framework for systematic pattern exploration."""

from .params import ParameterGrid, ParameterSpace
from .runner import ExperimentRunner

__all__ = ["ParameterSpace", "ParameterGrid", "ExperimentRunner"]
