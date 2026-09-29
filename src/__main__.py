"""Entry point for ``uv run python -m src <command>``."""
import sys

import fire
from pydantic import ValidationError

from .commands import RagCLI
from .exceptions import ArgError, RagError


def main() -> None:
    """Run the command-line interface."""
    try:
        try:
            fire.Fire(RagCLI)
        except ValidationError as e:
            raise ArgError(e) from e
    except RagError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
