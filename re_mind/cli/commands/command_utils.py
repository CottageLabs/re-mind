import re


def extract_command_arg(command_prefix: str, user_input: str) -> str:
    """Extract argument from command input by removing the command prefix."""
    return re.sub(rf'^{re.escape(command_prefix)}\b\s*', '', user_input).strip()