"""

Åshild Telle / Simula Research Labratory / 2019

"""

import os

import numpy as np
import matplotlib.pyplot as plt

class Metric:
    def __init__(self, maxima):
        self.maxima = maxima
        self.maximum = max(maxima)

    def calc_metric_value(self):
        pass

    def plot_metric_time(self, time, disp, maxima, path, mark_maxima=False):
        pass


class Beatrate(Metric):
    def __init__(self, disp_time, maxima):
        self.disp_time = disp_time
        super().__init__(maxima)

    def get_header(self):
        return "Beatrate"

    def calc_metric_value(self):
        """

        Calculates average of metric data at given maximum indices.

        Arguments:
            maxima - list of indices

        Returns:
            Average beatrate

        """

        maxima = self.maxima

        return np.mean(np.array([(maxima[k] - maxima[k-1]) \
                        for k in range(1, len(maxima))]))

    def plot_metric_time(self, time, path, mark_maxima=False):
        disp = self.disp_time
        maxima = self.maxima

        plt.plot(time, disp)

        # maxima
        for t in [time[m] for m in maxima]:
            plt.axvline(x=t, color='red')

        # visual properties
        plt.xlabel('Time (s)')
        plt.ylabel('Average displacement ($\mu m$)')
        plt.title(self.get_header())

        # save as ...
        filename = os.path.join(path, "beatrate_time.png")
        plt.savefig(filename, dpi=1000)

        plt.clf()