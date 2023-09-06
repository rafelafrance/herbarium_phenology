"""Test the term matchers."""
import unittest

from tests.setup import test


class TestTerm(unittest.TestCase):
    """Test the flowering trait parser."""

    def test_term_01(self):
        self.assertEqual(
            test("flowering"),
            [{"end": 9, "start": 0, "trait": "flowering", "flowering": "flowering"}],
        )

    def test_term_02(self):
        self.assertEqual(
            test("""Flower: N Fruit: N Vegetation: N Bud: N"""),
            [
                {
                    "not_flowering": "Flower: N",
                    "trait": "not_flowering",
                    "start": 0,
                    "end": 9,
                },
                {
                    "not_fruiting": "Fruit: N",
                    "trait": "not_fruiting",
                    "start": 10,
                    "end": 18,
                },
                {
                    "not_leaf_out": "Vegetation: N",
                    "trait": "not_leaf_out",
                    "start": 19,
                    "end": 32,
                },
            ],
        )

    def test_term_03(self):
        self.assertEqual(
            test("flowering/fruiting"),
            [
                {"flowering": "flowering", "trait": "flowering", "start": 0, "end": 9},
                {"fruiting": "fruiting", "trait": "fruiting", "start": 10, "end": 18},
            ],
        )
