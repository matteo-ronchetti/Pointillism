import bisect
import scipy.spatial
import numpy as np
import random
from .utils import regulate, limit_size, clipped_addition
from .vector_field import VectorField
from .color_palette import ColorPalette


def compute_color_probabilities(pixels, palette, k=9):
    distances = scipy.spatial.distance.cdist(pixels, palette.colors)
    maxima = np.amax(distances, axis=1)

    distances = maxima[:, None] - distances
    summ = np.sum(distances, 1)
    distances /= summ[:, None]

    distances = np.exp(k*len(palette)*distances)
    summ = np.sum(distances, 1)
    distances /= summ[:, None]

    return np.cumsum(distances, axis=1, dtype=np.float32)


def color_select(probabilities, palette):
    r = random.uniform(0, 1)
    i = bisect.bisect_left(probabilities, r)
    return palette[i] if i < len(palette) else palette[-1]


def randomized_grid(h, w, scale):
    assert (scale > 0)

    r = scale//2

    grid = []
    for i in range(0, h, scale):
        for j in range(0, w, scale):
            y = random.randint(-r, r) + i
            x = random.randint(-r, r) + j

            grid.append((y % h, x % w))

    random.shuffle(grid)
    return grid

