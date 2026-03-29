import numpy as np

from bp_designs.core.field import ConstantField
from bp_designs.core.geometry import Canvas, Point, Polygon
from bp_designs.core.pattern import Pattern
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.branching.strategies import FieldInfluenceGrowth


class MockPointPattern(Pattern):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def to_geometry(self, canvas=None):
        return Point(self.x, self.y, None)
    def render(self, context, style=None, **kwargs): pass
    def __str__(self): return "MockPoint"

class MockPolygonPattern(Pattern):
    def __init__(self, coords):
        self.coords = np.array(coords)
    def to_geometry(self, canvas=None):
        return Polygon(coords=self.coords)
    def render(self, context, style=None, **kwargs): pass
    def __str__(self): return "MockPolygon"

def test_field_influence_growth():
    canvas = Canvas.from_width_height(100, 100)
    root = MockPointPattern(50, 0)
    # Boundary is a box from (0, 0) to (100, 100)
    boundary = MockPolygonPattern([[0, 0], [100, 0], [100, 100], [0, 100]])

    # Constant field pointing right
    field = ConstantField(np.array([1.0, 0.0]))
    growth_strategy = FieldInfluenceGrowth(field, weight=1.0, segment_length=2.0)

    gen = SpaceColonization(
        canvas=canvas,
        root_position=root,
        initial_boundary=boundary,
        final_boundary=boundary,
        num_attractions=10, # Small number for quick test
        growth_strategy=growth_strategy,
        max_iterations=5
    )

    network = gen.generate_pattern()

    # With weight=1.0, all growth should be exactly [2.0, 0.0]
    # Starting at (50, 0), next nodes should be (52, 0), (54, 0), etc.
    # (Assuming they grow)

    # Let's check if the average x-coordinate of nodes (excluding root) is > 50
    positions = network.positions
    if len(positions) > 1:
        avg_x = np.mean(positions[1:, 0])
        assert avg_x > 50.0
