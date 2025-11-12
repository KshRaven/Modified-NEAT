
import pygame


class Window(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.surface = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pong")

    @property
    def shape(self):
        return self.width, self.height