"""Tests for Experiment Runner."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from bp_designs.core.geometry import Polyline
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner


@pytest.fixture
def temp_experiment_dirs(tmp_path):
    """Return temporary output and gallery directories."""
    output_dir = tmp_path / "experiments"
    gallery_dir = tmp_path / "gallery"
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, gallery_dir


@pytest.fixture
def experiment_runner_factory(temp_experiment_dirs):
    """Return a factory function for creating ExperimentRunner instances.

    The factory automatically uses temporary directories unless overridden.
    """
    output_dir, gallery_dir = temp_experiment_dirs

    def factory(**kwargs):
        # Set default directories if not provided
        if "output_dir" not in kwargs:
            kwargs["output_dir"] = output_dir
        if "gallery_dir" not in kwargs:
            kwargs["gallery_dir"] = gallery_dir
        return ExperimentRunner(**kwargs)

    return factory


class TestExperimentRunner:
    """Test Experiment Runner."""

    def setup_method(self):
        """Setup method-level fixtures."""
        # Simple mock geometry for testing
        self.mock_geometry = Polyline(polylines=[np.array([[0, 0], [10, 10], [20, 0]])])

    def test_initialization(self, experiment_runner_factory, temp_experiment_dirs):
        """Test ExperimentRunner initialization and directory creation."""
        output_dir, gallery_dir = temp_experiment_dirs
        runner = experiment_runner_factory(
            experiment_name="test_experiment",
            svg_width=200,
            svg_height=150,
            stroke_width=0.5,
        )

        assert runner.experiment_name == "test_experiment"
        assert runner.output_dir == output_dir
        assert runner.svg_width == 200
        assert runner.svg_height == 150
        assert runner.stroke_width == 0.5
        assert runner.gallery_dir == gallery_dir

        # Check that experiment directory was created
        exp_dir = output_dir / "test_experiment"
        outputs_dir = exp_dir / "outputs"
        assert exp_dir.exists()
        assert outputs_dir.exists()

    def test_parameter_grid_creation(self):
        """Test ParameterGrid creation and iteration."""
        # Create a simple parameter space
        space = ParameterSpace(
            name="test_space",
            ranges={
                "param1": [1, 2, 3],
                "param2": (0.0, 1.0, 3),  # Linear spacing
            },
            fixed={"fixed_param": "constant"},
        )

        grid = space.to_grid()

        assert grid.space_name == "test_space"
        assert len(grid) == 9  # 3 × 3 combinations
        assert set(grid.param_names) == {"param1", "param2"}
        assert grid.fixed_params == {"fixed_param": "constant"}

        # Test iteration
        combinations = list(grid)
        assert len(combinations) == 9

        # Check that each combination has all parameters
        for combo in combinations:
            assert "param1" in combo
            assert "param2" in combo
            assert "fixed_param" in combo
            assert combo["fixed_param"] == "constant"

        # Test indexing
        assert grid[0] == combinations[0]
        assert grid[8] == combinations[8]

        # Test summary generation
        summary = grid.summary()
        assert "test_space" in summary
        assert "9" in summary  # Number of combinations
        assert "param1" in summary
        assert "param2" in summary
        assert "fixed_param" in summary

    def test_run_experiment_success(self, experiment_runner_factory, temp_experiment_dirs):
        """Test running experiment with successful generations."""
        output_dir, gallery_dir = temp_experiment_dirs
        # Create a simple parameter grid with geometry objects
        # This tests that geometry objects can be serialized in params
        import numpy as np

        from bp_designs.core.geometry import Canvas, Point, Polygon

        # Create geometry objects
        canvas = Canvas(coords=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
        root_position = Point(x=50, y=50, z=None)
        boundary = Polygon(coords=np.array([[10, 10], [90, 10], [90, 90], [10, 90]]))

        space = ParameterSpace(
            name="test_run",
            ranges={"seed": [1, 2, 3]},
            fixed={
                "size": 100,
                "canvas": canvas,
                "root_position": root_position,
                "boundary": boundary,
            },
        )
        grid = space.to_grid()

        # Mock generator function that returns geometry
        def mock_generator(params):
            # Return a simple geometry based on seed
            seed = params["seed"]
            np.random.seed(seed)
            points = np.random.rand(3, 2) * params["size"]
            return Polyline(polylines=[points])

        runner = experiment_runner_factory(
            experiment_name="test_run",
            svg_width=100,
            svg_height=100,
            stroke_width=0.3,
        )

        # Run experiment with max_variants to limit test time
        summary = runner.run(grid, mock_generator, max_variants=2)

        # Check summary
        assert summary["experiment_name"] == "test_run"
        assert summary["parameter_space"] == "test_run"
        assert summary["total_variants"] == 2
        assert summary["successful"] == 2
        assert summary["failed"] == 0
        assert summary["elapsed_seconds"] >= 0
        assert "timestamp" in summary

        # Check that outputs were created
        exp_dir = output_dir / "test_run"
        outputs_dir = exp_dir / "outputs"

        assert (outputs_dir / "var_0001.svg").exists()
        assert (outputs_dir / "var_0001.json").exists()
        assert (outputs_dir / "var_0002.svg").exists()
        assert (outputs_dir / "var_0002.json").exists()

        # Check metadata files
        with open(outputs_dir / "var_0001.json") as f:
            metadata = json.load(f)
            assert metadata["variant_id"] == "var_0001"
            assert metadata["params"]["seed"] == 1
            assert metadata["params"]["size"] == 100
            assert metadata["svg_path"] == "outputs/var_0001.svg"
            assert metadata["svg_size"]["width"] == 100
            assert metadata["svg_size"]["height"] == 100

            # Check that geometry objects were serialized
            params = metadata["params"]
            assert "canvas" in params
            assert "root_position" in params
            assert "boundary" in params
            assert isinstance(params["canvas"], dict)
            assert isinstance(params["root_position"], dict)
            assert isinstance(params["boundary"], dict)
            assert "coords" in params["canvas"]
            assert "x" in params["root_position"] and "y" in params["root_position"]
            assert "coords" in params["boundary"]

        # Check config file
        config_path = exp_dir / "config.json"
        assert config_path.exists()
        with open(config_path) as f:
            config = json.load(f)
            assert config["experiment_name"] == "test_run"
            assert config["successful"] == 2

    def test_run_experiment_with_failures(self, experiment_runner_factory, temp_experiment_dirs):
        """Test running experiment with some failing generations."""
        output_dir, gallery_dir = temp_experiment_dirs
        # Create parameter grid
        space = ParameterSpace(
            name="test_failures",
            ranges={"mode": ["success", "fail"]},
        )
        grid = space.to_grid()

        # Mock generator that fails for specific mode
        def mock_generator(params):
            if params["mode"] == "fail":
                raise ValueError("Simulated failure")
            return Polyline(polylines=[np.array([[0, 0], [10, 10]])])

        runner = experiment_runner_factory(
            experiment_name="test_failures",
        )

        summary = runner.run(grid, mock_generator)

        # Check summary
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert len(summary["failures"]) == 1
        assert summary["failures"][0]["variant_id"] == "var_0002"
        assert summary["failures"][0]["params"]["mode"] == "fail"
        assert "Simulated failure" in summary["failures"][0]["error"]

        # Only successful variant should have output files
        exp_dir = output_dir / "test_failures"
        outputs_dir = exp_dir / "outputs"

        assert (outputs_dir / "var_0001.svg").exists()
        assert (outputs_dir / "var_0001.json").exists()
        assert not (outputs_dir / "var_0002.svg").exists()
        assert not (outputs_dir / "var_0002.json").exists()

    def test_run_experiment_with_dict_result(self, experiment_runner_factory, temp_experiment_dirs):
        """Test generator function returning dict with geometry and metadata."""

        output_dir, gallery_dir = temp_experiment_dirs

        space = ParameterSpace(
            name="test_dict_result",
            ranges={"value": [1, 2]},
        )
        grid = space.to_grid()

        def mock_generator(params):
            value = params["value"]
            geometry = Polyline(polylines=[np.array([[0, 0], [value * 10, value * 10]])])
            return {
                "geometry": geometry,
                "metadata": {"computed_value": value * 2},
                "extra_info": "test",
            }

        runner = experiment_runner_factory(experiment_name="test_dict_result")

        _summary = runner.run(grid, mock_generator, max_variants=1)

        # Check that metadata was saved
        exp_dir = output_dir / "test_dict_result"
        outputs_dir = exp_dir / "outputs"
        with open(outputs_dir / "var_0001.json") as f:
            metadata = json.load(f)
            assert metadata["metadata"]["computed_value"] == 2  # 1 * 2
            assert metadata["extra_info"] == "test"
            assert "geometry" not in metadata  # Should be excluded

    def test_run_experiment_with_numpy_metadata(
        self, experiment_runner_factory, temp_experiment_dirs
    ):
        """Test handling of numpy arrays in metadata."""

        output_dir, gallery_dir = temp_experiment_dirs

        space = ParameterSpace(
            name="test_numpy_metadata",
            ranges={"id": [1]},
        )
        grid = space.to_grid()

        def mock_generator(params):
            geometry = Polyline(polylines=[np.array([[0, 0], [10, 10]])])
            return {
                "geometry": geometry,
                "array_data": np.array([1, 2, 3]),
                "scalar_data": np.float64(42.5),
            }

        runner = experiment_runner_factory(experiment_name="test_numpy_metadata")

        _summary = runner.run(grid, mock_generator)

        # Check that numpy arrays were converted to lists
        exp_dir = output_dir / "test_numpy_metadata"
        outputs_dir = exp_dir / "outputs"
        with open(outputs_dir / "var_0001.json") as f:
            metadata = json.load(f)
            assert metadata["array_data"] == [1, 2, 3]
            assert metadata["scalar_data"] == 42.5

    @patch.object(ExperimentRunner, "_update_experiments_index")
    def test_experiment_index_update(self, mock_update_index, experiment_runner_factory):
        """Test that experiments index is updated."""
        space = ParameterSpace(name="test_index", ranges={"x": [1]})
        grid = space.to_grid()

        runner = experiment_runner_factory(
            experiment_name="test_index",
        )

        def mock_generator(params):
            return Polyline(polylines=[np.array([[0, 0], [10, 10]])])

        summary = runner.run(grid, mock_generator)

        # Check that _update_experiments_index was called
        mock_update_index.assert_called_once()

        # Check it was called with the summary
        call_args = mock_update_index.call_args
        assert call_args[0][0] == summary

    def test_max_variants_limit(self, experiment_runner_factory, temp_experiment_dirs):
        """Test max_variants parameter limits number of generated variants."""

        output_dir, gallery_dir = temp_experiment_dirs

        space = ParameterSpace(
            name="test_max_variants",
            ranges={"x": list(range(10))},  # 10 values
        )
        grid = space.to_grid()

        call_count = 0

        def mock_generator(params):
            nonlocal call_count
            call_count += 1
            return Polyline(polylines=[np.array([[0, 0], [10, 10]])])

        runner = experiment_runner_factory(experiment_name="test_max_variants")

        # Limit to 3 variants
        summary = runner.run(grid, mock_generator, max_variants=3)

        assert summary["total_variants"] == 3
        assert summary["successful"] == 3
        assert call_count == 3

        # Only 3 output files should exist
        exp_dir = output_dir / "test_max_variants"
        outputs_dir = exp_dir / "outputs"
        svg_files = list(outputs_dir.glob("*.svg"))
        assert len(svg_files) == 3

    def test_empty_parameter_grid(self, experiment_runner_factory):
        """Test handling of empty parameter grid."""
        # Create empty parameter space
        space = ParameterSpace(name="test_empty", ranges={})
        grid = space.to_grid()

        # Empty ranges produce one combination (empty parameter set)
        assert len(grid) == 1
        assert list(grid) == [{}]

        runner = experiment_runner_factory(experiment_name="test_empty")

        def mock_generator(params):
            return Polyline(polylines=[np.array([[0, 0], [10, 10]])])

        # Should generate one variant with empty params
        summary = runner.run(grid, mock_generator)

        assert summary["total_variants"] == 1
        assert summary["successful"] == 1
        assert summary["failed"] == 0

    def test_generator_returns_non_geometry(self, experiment_runner_factory):
        """Test error when generator doesn't return geometry."""
        space = ParameterSpace(name="test_bad_return", ranges={"x": [1]})
        grid = space.to_grid()

        runner = experiment_runner_factory(experiment_name="test_bad_return")

        def bad_generator(params):
            return "not a geometry"  # Wrong return type

        summary = runner.run(grid, bad_generator)

        # Should record failure
        assert summary["successful"] == 0
        assert summary["failed"] == 1
        assert (
            "to_svg" in summary["failures"][0]["error"]
        )  # AttributeError about missing to_svg method
