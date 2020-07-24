import pytest
from gbpy.util_funcs import compute_ovito_data


def test_compute_ovito_data():
    filename0 = "./data/dump_1"
    data = compute_ovito_data(filename0)
    actual = 5765
    expected = data.particles.count
    assert actual == expected
