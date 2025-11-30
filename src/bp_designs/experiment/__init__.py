"""Experimentation framework for systematic pattern exploration."""

from bp_designs.experiment.gallery import generate_gallery
from bp_designs.experiment.params import ParameterGrid, ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner

__all__ = ["ParameterSpace", "ParameterGrid", "ExperimentRunner", "generate_gallery"]
