"""Smoke test to verify the package imports correctly."""

import music_genre


def test_package_imports() -> None:
    """The package must be importable."""
    assert music_genre is not None
