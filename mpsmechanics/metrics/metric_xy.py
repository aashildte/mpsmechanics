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
    def __init__(self, metric_data, e_alpha):
        if(e_alpha == None):
            self.metric_data = metric_data
        else:
            e_dir, label, head = e_alpha 
            
            self.metric_data = \
                an.calc_projection_vectors(metric_data, \
                     e_dir, over_time=True)
            

            self.header += ", " + head
            self.label += "_" + label
    
        self.over_time = self.calc_over_time()

    def over_time(self):
        pass

    def get_header(self):
        return self.header

    def get_ylabel(self):
        return self.ylabel

    def get_label(self):
        return self.label

    def calc_metric_value(self, maxima):
        """

        Calculates average of metric data at given maximum indices.

        Arguments:
            maxima - list of indices
        
        Returns:
            Average at peaks

        """         
        return np.mean([self.over_time[m] for m in maxima])


    def plot_metric_time(self, time, disp, maxima, path, mark_maxima=False):
 
        values = self.over_time
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


class Displacement(Metric_xy):
    def __init__(self, disp_data, e_alpha):
        self.header = "Displacement"
        self.ylabel = "Displacement (um)"
        self.label = "displacement"

        super().__init__(disp_data, e_alpha)

    def calc_over_time(self):
        return op.calc_norm_over_time(self.metric_data) 

class Velocity(Metric_xy):
    def __init__(self, disp_data, e_alpha):
        self.header = "Velocity"
        self.ylabel = "Velocity (um/s)"
        self.label = "velocity"

        velocity = np.gradient(disp_data, axis=0)
        super().__init__(velocity, e_alpha)

    def calc_over_time(self):
        return op.calc_norm_over_time(self.metric_data) 


class Principal_strain(Metric_xy):
    def __init__(self, disp_data, e_alpha):

         self.header = "Principal strain"
         self.ylabel = "Strain (-)"
         self.label = "principal_strain"

         pr_strain = mc.calc_principal_strain(disp_data, \
            over_time=True)
         super().__init__(pr_strain, e_alpha)

    def calc_over_time(self):
        return op.calc_norm_over_time(self.metric_data) 


class Prevalence(Metric_xy):
    def __init__(self, disp_data, threshold, e_alpha): 

         self.header = "Prevalence"
         self.ylabel = "Prevalence (-)"
         self.label = "prevalence"

         prev_xy = mc.calc_prevalence(disp_data, threshold)
         
         super().__init__(prev_xy, e_alpha)
 
    def calc_over_time(self):
        return np.sum(self.metric_data, axis=(1, 2)) 

