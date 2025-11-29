"""
Basic package tests to verify setup and imports.
"""

import bp_designs


class TestPackage:
    """Test basic package functionality."""

    def test_package_import(self):
        """Test that the package can be imported."""
        assert bp_designs is not None

    def test_version_exists(self):
        """Test that version is defined."""
        assert hasattr(bp_designs, "__version__")
        assert bp_designs.__version__ == "0.1.0"
