#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Luke Hindman
"""
TEMPO Regression Test Runner

Runs the full TEMPO test suite and reports results for each test.

Usage:
    python run_tests.py              # run all tests
    python run_tests.py -k encode    # filter by keyword
    python run_tests.py -x           # stop on first failure
    python run_tests.py --no-header  # suppress the banner
"""

import argparse
import subprocess
import sys
from pathlib import Path


BANNER = """\
╔══════════════════════════════════════════════════════════════╗
║          TEMPO v1.1 — Regression Test Suite                  ║
╚══════════════════════════════════════════════════════════════╝
"""

TEST_MODULES = [
    ("Morse Table",       "tests/test_morse_table.py"),
    ("encode_word / split_channels", "tests/test_encode_word.py"),
    ("generate_dataset",  "tests/test_generate_dataset.py"),
    ("write_hdf5",        "tests/test_hdf5.py"),
    ("TEMPODataset",      "tests/test_tempo_dataset.py"),
    ("Protocol Invariants", "tests/test_protocol.py"),
    ("CLI",               "tests/test_cli.py"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-k', '--keyword', default=None,
                        help='Only run tests matching this keyword expression')
    parser.add_argument('-x', '--exitfirst', action='store_true',
                        help='Stop after the first failure')
    parser.add_argument('--no-header', action='store_true',
                        help='Skip the banner')
    parser.add_argument('--tb', default='short',
                        choices=['short', 'long', 'no', 'line', 'auto'],
                        help='Traceback style on failure (default: short)')
    args = parser.parse_args()

    if not args.no_header:
        print(BANNER)

    root = Path(__file__).parent

    pytest_args = [
        sys.executable, '-m', 'pytest',
        'tests/',
        '--verbose',                      # one line per test
        f'--tb={args.tb}',
        '--color=yes',
        '-p', 'no:cacheprovider',         # no .pytest_cache noise
    ]

    if args.keyword:
        pytest_args += ['-k', args.keyword]
    if args.exitfirst:
        pytest_args.append('-x')

    result = subprocess.run(pytest_args, cwd=str(root))
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
