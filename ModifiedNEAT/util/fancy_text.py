
from typing import Any

import colorama
from colorama import Fore, Style


def CM(text: Any, color: colorama.Fore = Fore.GREEN, style: colorama.Style = Style.BRIGHT):
    text: str = color + style + str(text) + Style.RESET_ALL
    return text
