"""

Åshild Telle / Simula Research Labratory / 2019

"""

import os

import numpy as np
import matplotlib.pyplot as plt

from ..dothemaths import mechanical_properties as mc
from ..dothemaths import operations as op
from ..dothemaths import angular as an
from ..dothemaths import heartbeat as hb

from .metric import Metric

class Metric_xy(Metric):
    def __init__(self, metric_data, e_alpha, movement, maxima):
        
        super().__init__(maxima)

        if(e_alpha == None):
            self.projected = False
            self.metric_data = metric_data
            self.projection_label = "norm"
        else:
            self.projected = True

            e_dir, label, head = e_alpha 
            
            self.metric_data = \
                an.calc_projection_vectors(metric_data, \
                     e_dir, over_time=True)
            

            self.header += ", " + head
            self.projection_label = label
  
        self.movement = movement 
        self.over_time = self.calc_over_time() 

    def over_time(self):
        pass

    def get_header(self):
        return self.header

    def get_ylabel(self):
        return self.ylabel

    def get_label(self):
        return self.label + "_" + self.get_projection_label()

    def get_projection_label(self):
        return self.projection_label

    def calc_metric_value(self):
        """

        Calculates average of metric data at given maximum indices.

        TODO add std? other values?

        Arguments:
            maxima - list of indices
        
        Returns:
            Average at peaks

        """         
        return np.mean([self.over_time[m] for m in self.maxima])


    def plot_metric_time(self, time, path, mark_maxima=False):
 
        values = self.over_time
        maxima = self.maxima

        plt.plot(time, values)
        
        # maxima
        if(mark_maxima):
            m_t = [t[m] for m in maxima]
            max_vals = [values[m] for m in maxima]
        
            plt.scatter(m_t, max_vals, color='red')

        # visual properties
        plt.xlabel('Time (s)')
        plt.ylabel(self.get_ylabel())
        plt.title(self.get_header())
 
        # save as ...
        filename = os.path.join(path, self.get_label() + ".png")
        plt.savefig(filename, dpi=1000)

        plt.clf()

    
    def plot_spacial_dist(self, dimensions, path, over_time):

        # over_time is time-consuming yet possibly interesting;
        # having it as an option allows us to make movies
        # which might be useful, especially for presentations etc.

        data = self.metric_data
        maximum = self.maximum

        T = data.shape[0]

        if(over_time):
            for t in range(T):
                self._plot_vector_field_step(data[t], \
                        dimensions, path, t)
        else:
            self._plot_vector_field_step(data[maximum], \
                    dimensions, path, maximum)


    def _plot_vector_field_step(self, data, dimensions, path, t, norm=None):

        label = self.get_label()
        
        if(self.projected):
            f_txt = "magnitude_" + label + "_%04d.png" %t

            filename = os.path.join(path, f_txt)
            pv.plot_magnitude(data, dimensions, filename, norm)

        else:
            m_txt = "magnitude_" + label + "_%04d.png" %t
            d_txt = "direction_" + label + "_%04d.png" %t
            q_txt = "quiver_" + label + "_%04d.png" %t

            filenames = [os.path.join(path, txt) \
                    for txt in [m_txt, d_txt, q_txt]]

            pv.plot_magnitude(data, dimensions, filenames[0], norm)
            pv.plot_direction(data, dimensions, filenames[1])
            pv.plot_vector_field(data, dimensions, filenames[2])


class Displacement(Metric_xy):
    def __init__(self, disp_data, e_alpha, movement, maxima):
        self.header = "Displacement"
        self.ylabel = "Average displacement ($\mu m$)"
        self.label = "displacement"

        super().__init__(disp_data, e_alpha, movement, maxima)

    def calc_over_time(self):
        return op.calc_norm_over_time(self.metric_data, self.movement) 

class Velocity(Metric_xy):
    def __init__(self, disp_data, e_alpha, movement, maxima):
        self.header = "Velocity"
        self.ylabel = "Average velocity ($\mu m/s$)"
        self.label = "velocity"

        velocity = np.gradient(disp_data, axis=0)
        super().__init__(velocity, e_alpha, movement, maxima)

    def calc_over_time(self):
        return op.calc_norm_over_time(self.metric_data, self.movement) 


class Principal_strain(Metric_xy):
    def __init__(self, disp_data, e_alpha, movement, maxima):

        self.header = "Principal strain"
        self.ylabel = "Average strain (-)"
        self.label = "principal_strain"

        pr_strain = mc.calc_principal_strain(disp_data, \
            over_time=True)
        super().__init__(pr_strain, e_alpha, movement, maxima)

    def calc_over_time(self):
        return op.calc_norm_over_time(self.metric_data, self.movement) 


class Prevalence(Metric_xy):
    def __init__(self, disp_data, threshold, e_alpha, movement, maxima):

        self.header = "Prevalence"
        self.ylabel = "Prevalence (-)"
        self.label = "prevalence"

        prev_xy = mc.calc_prevalence(disp_data, threshold)
         
        super().__init__(prev_xy, e_alpha, movement, maxima)
 
    def calc_over_time(self):
        # Q: how do we scale prevalence?

        scale = 1./np.sum(np.sum(self.movement))
        #_, X, Y = self.metric_data.shape
        #scale = 1./(X*Y)
 
        return scale*np.sum(self.metric_data, axis=(1, 2))

