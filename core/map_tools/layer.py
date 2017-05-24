#!/usr/bin/env python
"""Provides a common base class for all map layers.

Since many layers share parameters and functions, the ``layer`` module
defines these in one place, allowing all layers to use it as a
superclass.

"""
__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Stable"

from abc import ABCMeta, abstractmethod
import logging
import matplotlib.pyplot as plt

# TODO: @Refactor Is this necessary? Especially the self.target


class Layer(object):
    __metaclass__ = ABCMeta
    """A collection of generic layer parameters and functions.

    .. image:: img/classes_Layer.png

    Parameters
    ----------
    bounds : array_like, optional
        Map boundaries as [xmin,ymin,xmax,ymax] in [m]. Defaults to
        [-5, -5, 5, 5].
    visible : bool, optional
        Whether or not the layer is shown when plotting.
        To be used for toggling layers.
    ax : axes handle, optional
        The axes to be used for plotting. Defaults to current axes.
    alpha : float, optional
        The layer's transparency, from 0 to 1. Defaults to 0.8.
    cmap_str : str, optional
        The colormap string for the layer. Defaults to `'jet'`.

    """
    def __init__(self, bounds=[-5, -5, 5, 5], visible=True,
                 fig=None, ax=None, alpha=0.8, cmap_str='jet'):
        if fig is None:
            fig = plt.gcf()
        self.fig = fig
        if ax is None:
            ax = plt.gca()
        self.ax = ax

        self.bounds = bounds  # [xmin,ymin,xmax,ymax] in [m]
        self.visible = visible
        self.alpha = alpha
        self.cmap = plt.cm.get_cmap(cmap_str)

    @abstractmethod
    def plot(self):
        """Plots the layer on the axis.

        This function adds a new layer to the axis. It often takes parameters,
        such as a dictionary of shapes, or a probability distribution.
        Matplotlib is the go to plotting module.

        It is generally used in the update function for animations,
        but can also be used to plot static images for testing or
        other purposes.
        """
        pass

    @abstractmethod
    def remove(self):
        """Removes the layer from the axis.

        This function must be able to remove everything that plot adds. It
        may have logic to leave certain parts on, as is in the case of the
        shape_layer.

        It is generally used in the update function for animations,
        but can also be used to remove a layer entirely.
        """
        pass

    @abstractmethod
    def update(self, i=0):
        """Provides the animation update function for this layer

        This function generally calls remove() first to remove some or
        all of the previously plotted objects. Then plot() adds the new or
        updated objects.

        i counts from 0 at the start of the animation, and can be used to
        schedule certain things.
        """
        pass
