"""Check a command line against a Fire component before Fire runs it.

Fire calls the command with the arguments it understands and only then
complains about the rest, so a typo in an option (``--chunk_size``), too
many arguments, or a ``--help`` after other arguments would run the whole
command first. ``ArgvGuard`` catches those cases up front.
"""
import inspect
from typing import Any

from .exceptions import UsageError


class ArgvGuard:
    """Validate ``argv`` against the public methods of a Fire component."""

    SEPARATORS: frozenset[str] = frozenset({"--", "-"})
    HELP_FLAGS: frozenset[str] = frozenset({"--help", "-help", "-h"})

    def __init__(self, component: type[Any]) -> None:
        """Keep the class whose methods are the commands.

        Args:
            component: The class given to ``fire.Fire``.
        """
        self._component = component

    def check(self, argv: list[str]) -> list[str]:
        """Return the command line Fire should receive.

        Args:
            argv: Arguments after the program name.

        Returns:
            ``argv`` unchanged, or ``[command, "--", "--help"]`` when help
            was asked for, so the command is never run.

        Raises:
            UsageError: Unknown option or too many arguments.
        """
        if not argv or argv[0] not in self._commands():
            return argv
        command, args = argv[0], self._before_separator(argv[1:])
        if any(arg in self.HELP_FLAGS for arg in args):
            return [command, "--", "--help"]
        params = self._params(command)
        named, positional = self._split(command, args, params)
        if len(named) + positional > len(params):
            raise UsageError(
                f"Too many arguments for '{command}': it takes at most "
                f"{len(params)} ({self._options(params)})"
            )
        return argv

    def _commands(self) -> set[str]:
        """Names of the public methods (the available commands)."""
        return {
            name for name, _ in inspect.getmembers(self._component,
                                                   inspect.isfunction)
            if not name.startswith("_")
        }

    def _params(self, command: str) -> list[str]:
        """Parameter names of a command, without ``self``."""
        signature = inspect.signature(getattr(self._component, command))
        return [name for name in signature.parameters if name != "self"]

    def _before_separator(self, args: list[str]) -> list[str]:
        """Arguments up to the first separator (the rest is for Fire)."""
        for position, arg in enumerate(args):
            if arg in self.SEPARATORS:
                return args[:position]
        return args

    def _split(self, command: str, args: list[str],
               params: list[str]) -> tuple[set[str], int]:
        """Resolve the options and count the positional arguments.

        Returns:
            The parameters given as options and the number of positional
            arguments.

        Raises:
            UsageError: An option does not match any parameter, or a
                non-boolean option has no value.
        """
        named: set[str] = set()
        positional = 0
        position = 0
        while position < len(args):
            arg = args[position]
            if self._is_option(arg):
                name, has_value = self._option_name(arg)
                param = self._resolve(command, arg, name, params)
                named.add(param)
                if not has_value:
                    if self._value_follows(args, position):
                        position += 1
                    elif not self._is_bool(command, param):
                        raise UsageError(
                            f"Option '{arg}' of '{command}' needs a value"
                        )
            else:
                positional += 1
            position += 1
        return named, positional

    def _value_follows(self, args: list[str], position: int) -> bool:
        """Whether the argument after ``position`` can be its value."""
        return (position + 1 < len(args)
                and not self._is_option(args[position + 1]))

    def _is_bool(self, command: str, param: str) -> bool:
        """Whether a parameter is a flag Fire may set with no value.

        ``--verbose`` alone means ``True`` for a ``bool`` parameter.
        """
        signature = inspect.signature(getattr(self._component, command))
        return signature.parameters[param].annotation is bool

    def _is_option(self, arg: str) -> bool:
        """``-x``/``--x`` is an option; ``-3`` or ``-2.5`` is a value."""
        if not arg.startswith("-") or arg in self.SEPARATORS:
            return False
        try:
            float(arg)
        except ValueError:
            return True
        return False

    def _option_name(self, arg: str) -> tuple[str, bool]:
        """Parameter name written in an option and whether it has ``=``.

        ``--max-chunk-size=5`` → ``("max_chunk_size", True)``.
        """
        body = arg.lstrip("-")
        name, equals, _ = body.partition("=")
        return name.replace("-", "_"), bool(equals)

    def _resolve(self, command: str, arg: str, name: str,
                 params: list[str]) -> str:
        """Match an option to a parameter, as Fire does.

        Accepts the full name or, when it is unique, its first letter.

        Raises:
            UsageError: No parameter matches.
        """
        if name in params:
            return name
        if len(name) == 1:
            matches = [param for param in params if param.startswith(name)]
            if len(matches) == 1:
                return matches[0]
        raise UsageError(
            f"Unknown option '{arg}' for '{command}'. "
            f"Valid options: {self._options(params)}"
        )

    def _options(self, params: list[str]) -> str:
        """``--a, --b`` list for error messages."""
        return ", ".join(f"--{param}" for param in params) or "none"
