from app.ml.distributions import bracket_probability


def test_bracket_probabilities_are_coherent() -> None:
    below = bracket_probability(70, 2, None, 68)
    middle = bracket_probability(70, 2, 68, 72)
    above = bracket_probability(70, 2, 72, None)
    assert abs((below + middle + above) - 1.0) < 1e-4
    assert middle > below
    assert middle > above
