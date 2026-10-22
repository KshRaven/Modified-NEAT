
import pygame


class Window(object):
    def __init__(self, width: int, height: int, blob_size=10):
        self.width = (width + 2) * blob_size
        self.height = (height + 2) * blob_size
        self.blob_size = blob_size

        self.surface = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Env")

    @property
    def shape(self):
        return self.width, self.height