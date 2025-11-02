
from colorama import Fore, Style
from typing import Any

import colorama


def CM(text: Any, color: colorama.Fore = Fore.GREEN, style: colorama.Style = Style.BRIGHT):
    text: str = color + style + str(text) + Style.RESET_ALL
    return text
