"""CLI entrypoint."""
import argparse

from argparse import Namespace


def parse_args() -> Namespace:
    """Parse script args."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', '-f',
                        help="Filepath to use for ranking.")

    args = parser.parse_args()
    return args


def run() -> None:
    """Run the CLI."""
    args = parse_args()

    print("hlucb CLI running")
    if args.filepath is not None:
        print(f"Filepath: {args.filepath}")
