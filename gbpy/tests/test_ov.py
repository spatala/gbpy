import pytest
from gbpy.util_funcs import compute_ovito_data


def test_compute_ovito_data():
    filename0 = "tests/data/dump_1"
    data = compute_ovito_data(filename0)
    actual = 4590
    expected = data.particles.count
    assert actual == expected
