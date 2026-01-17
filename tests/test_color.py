from bp_designs.core.color import Color


def test_complementary_red():
    c = Color.from_hex("#ff0000")
    comp = c.complementary()
    # Red complement should be Green in RYB
    h, s, lightness = comp.to_hsl()
    # Green is around 0.33
    assert 0.3 <= h <= 0.36


def test_complementary_green():
    c = Color.from_hex("#00ff00")
    comp = c.complementary()
    # Green complement should be Red in RYB
    h, s, lightness = comp.to_hsl()
    # Red is 0.0 or 1.0
    assert h < 0.05 or h > 0.95


def test_complementary_blue():
    c = Color.from_hex("#0000ff")
    comp = c.complementary()
    # Blue complement should be Orange in RYB
    h, s, lightness = comp.to_hsl()
    # Orange is around 0.08 (30 degrees)
    assert 0.05 <= h <= 0.12


def test_complementary_yellow():
    c = Color.from_hex("#ffff00")
    comp = c.complementary()
    # Yellow complement should be Purple/Violet in RYB
    h, s, lightness = comp.to_hsl()
    # Violet is around 0.75-0.83
    assert 0.7 <= h <= 0.85
