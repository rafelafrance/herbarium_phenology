"""Test the term matchers."""
import unittest

from tests.setup import test


class TestTerm(unittest.TestCase):
    """Test the flowering trait parser."""

    def test_term_01(self):
        self.assertEqual(
            test("flowering"), [{"end": 9, "start": 0, "trait": "flowering"}]
        )

    def test_term_02(self):
        self.assertEqual(
            test("""Flower: N Fruit: N Vegetation: N Bud: N"""),
            [
                {"end": 9, "start": 0, "trait": "not_flowering"},
                {"end": 18, "start": 10, "trait": "not_fruiting"},
                {"end": 32, "start": 19, "trait": "not_leaf_out"},
            ],
        )

    def test_term_03(self):
        self.assertEqual(
            test("flowering/fruiting"),
            [
                {"end": 9, "start": 0, "trait": "flowering"},
                {"end": 18, "start": 10, "trait": "fruiting"},
            ]
        )
