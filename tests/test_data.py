"""Basic smoke tests for project data artifacts.

These are intentionally lightweight to keep the test suite fast while ensuring
core modules import correctly. Expand with more specific tests as the project grows.
"""

from src.labels import LABELS


def test_labels_non_empty():
    """LABELS should contain at least one class name."""
    assert isinstance(LABELS, list)
    assert len(LABELS) >= 1

