"""
CS131 - Computer Vision: Foundations and Applications
Project 2 Option B
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/25/2022
Python Version: 3.5+
"""

import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    for i in range(Hi):
        for j in range(Wi):
            window = padded[i:i+Hk, j:j+Wk]
            out[i, j] = np.sum(window * kernel)

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """
    Dx = np.array([[-1, 0, 1]]) / 2
    return conv(img, Dx)

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """
    Dy = np.array([[-1], [0], [1]]) / 2
    return conv(img, Dy)

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
        theta: Direction (in degrees, 0 <= theta < 360) of gradient.
    """
    Gx = partial_x(img)
    Gy = partial_y(img)

    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.degrees(np.arctan2(Gy, Gx))
    theta = (theta + 360) % 360

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)

    for i in range(1, H-1):
        for j in range(1, W-1):
            angle = theta[i, j]
            curr = G[i, j]

            if angle == 0 or angle == 180:
                n1 = G[i, j + 1]
                n2 = G[i, j - 1]
            elif angle == 45 or angle == 225:
                n1 = G[i - 1, j + 1]
                n2 = G[i + 1, j - 1]
            elif angle == 90 or angle == 270:
                n1 = G[i - 1, j]
                n2 = G[i + 1, j]
            else:
                n1 = G[i - 1, j - 1]
                n2 = G[i + 1, j + 1]

            if curr > n1 and curr > n2:
                out[i, j] = curr
            else:
                out[i, j] = 0

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold (float) for strong edges.
        low: low threshold (float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
        weak_edges: Boolean array representing weak edges.
    """
    strong_edges = img > high
    weak_edges = (img > low) & (img <= high)

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x). """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).

    Returns:
        edges: numpy boolean array of shape (H, W).
    """
    from collections import deque

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T

    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    queue = deque()
    for (y, x) in indices:
        queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for (i, j) in get_neighbors(y, x, H, W):
            if weak_edges[i, j]:
                edges[i, j] = True
                weak_edges[i, j] = False
                queue.append((i, j))

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=None, low=None):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: grayscale image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges (auto-computed if None).
        low: low threshold for weak edges (auto-computed if None).

    Returns:
        edge: numpy boolean array of shape (H, W).
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel / np.sum(kernel)

    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)

    # Adaptive thresholds based on image's gradient distribution
    if high is None:
        high = np.percentile(nms, 90)
    if low is None:
        low = high * 0.5

    strong, weak = double_thresholding(nms, high, low)
    edge = link_edges(strong, weak)

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Args:
        img: binary image of shape (H, W).

    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        for j in range(num_thetas):
            rho = x * cos_t[j] + y * sin_t[j]
            rho_idx = np.argmin(np.abs(rhos - rho))
            accumulator[rho_idx, j] += 1

    return accumulator, rhos, thetas