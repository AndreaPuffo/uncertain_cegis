import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

"""https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html"""

def plot_ellipse_matrix_form(matrix, ax, radius_ellipse=1, facecolor='none', **kwargs):
    """
    Create a plot of the ellipse given by *matrix*

    Parameters
    ----------
    matrix : array-like, shape (n, n)
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    A = matrix[0,0]
    B = matrix[0,1] + matrix[1,0]
    C = matrix[1,1]
    theta = 0.5 * np.arctan2(-B, C-A)
    # find the reasoning at https://en.wikipedia.org/wiki/Ellipse
    a = - np.sqrt( 2. * -radius_ellipse*(B**2 - 4.*A*C) * ( A + C + np.sqrt( (A-C)**2 + B**2 ) ) ) / (B**2 - 4.*A*C)
    b = - np.sqrt( 2. * -radius_ellipse*(B**2 - 4.*A*C) * ( A + C - np.sqrt( (A-C)**2 + B**2 ) ) ) / (B**2 - 4.*A*C)

    ellipse = Ellipse((0, 0), width=a * 2, height=b * 2, angle=np.rad2deg(theta),
                      facecolor=facecolor, **kwargs)

    ax.add_patch(ellipse)
    ax.autoscale_view()
