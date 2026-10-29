import sys

from .graph import run_triage


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('usage: python -m field_triage.cli "<fault report text>"')
    state = run_triage(sys.argv[1])
    print(state["final"].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
