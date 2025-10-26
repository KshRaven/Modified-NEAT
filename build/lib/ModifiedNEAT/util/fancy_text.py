
from colorama import Fore, Style

import colorama


def CM(text: str, color: colorama.Fore = Fore.GREEN, style: colorama.Style = Style.BRIGHT):
    text: str = color + style + str(text) + Style.RESET_ALL
    return text
