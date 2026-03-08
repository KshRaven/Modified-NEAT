import pygame


class Window(object):
    def __init__(self, width: int, height: int, headless: bool = False):
        self.width = width
        self.height = height
        self.headless = headless
        self.surface = None

        if not headless:
            self.surface = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Pong")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.width, self.height)

    def enable_display(self):
        """Enable display if running in headless mode"""
        if self.headless and self.surface is None:
            self.surface = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pong")
            self.headless = False

    def __repr__(self):
        return f"Window(width={self.width}, height={self.height}, headless={self.headless})"