"""
Regression tests for the MORSE_TABLE constant.

Verifies completeness (all A-Z, 0-9 present), structural validity
(only dots and dashes), and correctness of selected known codes.
"""

import pytest
from tempo.dataset.generate_dataset import MORSE_TABLE


class TestCompleteness:
    def test_all_uppercase_letters_present(self):
        for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            assert ch in MORSE_TABLE, f"'{ch}' missing from MORSE_TABLE"

    def test_all_digits_present(self):
        for ch in '0123456789':
            assert ch in MORSE_TABLE, f"'{ch}' missing from MORSE_TABLE"

    def test_total_entry_count(self):
        assert len(MORSE_TABLE) == 36  # 26 letters + 10 digits

    def test_no_lowercase_keys(self):
        for key in MORSE_TABLE:
            assert key == key.upper(), f"Unexpected lowercase key: '{key}'"


class TestStructuralValidity:
    def test_values_contain_only_dots_and_dashes(self):
        for char, code in MORSE_TABLE.items():
            bad = [c for c in code if c not in '.-']
            assert not bad, f"Invalid characters {bad!r} in code for '{char}': '{code}'"

    def test_all_codes_nonempty(self):
        for char, code in MORSE_TABLE.items():
            assert len(code) >= 1, f"Empty code for '{char}'"

    def test_longest_code_is_digits(self):
        """Digits (0-9) all have 5-element codes — the longest in standard Morse."""
        for ch in '0123456789':
            assert len(MORSE_TABLE[ch]) == 5, (
                f"Digit '{ch}' should have 5 elements, got {len(MORSE_TABLE[ch])}"
            )


class TestKnownCodes:
    """Spot-check internationally standardised Morse codes."""

    @pytest.mark.parametrize("char,expected", [
        ('E', '.'),
        ('T', '-'),
        ('I', '..'),
        ('M', '--'),
        ('S', '...'),
        ('O', '---'),
        ('A', '.-'),
        ('N', '-.'),
        ('R', '.-.'),
        ('K', '-.-'),
        ('H', '....'),
        ('B', '-...'),
        ('0', '-----'),
        ('1', '.----'),
        ('9', '----.'),
        ('5', '.....'),
    ])
    def test_known_code(self, char, expected):
        assert MORSE_TABLE[char] == expected, (
            f"Wrong code for '{char}': expected '{expected}', got '{MORSE_TABLE[char]}'"
        )
