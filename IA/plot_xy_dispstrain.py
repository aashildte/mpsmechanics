
import sys
import numpy as np
import matplotlib.colors as cl
import matplotlib.pyplot as plt

import operations as op
import preprocessing as pp
import mechanical_properties as mc
import plot_vector_field as pl
import io_funs as io
import angular as an

def read_values(f_in, idt, dimensions):

    xlen = 664.30*1E-6

    disp_data, scale = io.read_disp_file(f_in, xlen)

    alpha = 0.75
    N_d = 5
    
    time_step = op.calc_max_ind(op.calc_norm_over_time(disp_data))

    e_alpha, e_beta = an.calc_direction_vectors(disp_data, dimensions)

    disp_data_t = pp.do_diffusion(disp_data[time_step], alpha, N_d, \
            over_time=False)
    strain_data_t = mc.calc_principal_strain(disp_data_t, \
            over_time=False)
    
    return scale*disp_data_t, strain_data_t, e_alpha, e_beta


def rotate(v, angle):
    return np.array([np.cos(angle)*v[0] - np.sin(angle)*v[1],
        np.sin(angle)*v[0] + np.cos(angle)*v[1]])

def plot_angle_distribution(data, e_alpha, p_value, p_label, path, title, idt):
    X, Y = data.shape[:2]

    values = []

    count1 = 0
    count2 = 0

    for x in range(X):
        for y in range(Y):
            norm = np.linalg.norm(data[x, y])
            if(norm > p_value and norm > 1E-14):
                ip = np.dot(data[x, y], e_alpha)/norm
                ip = max(min(ip, 1), -1)      # eliminate small overflow - due to rounding errors (?)
                angle = np.arccos(ip)
                values.append(angle)
                count1 = count1+1
                pm = 0.436332313   # 25 degrees
                if(angle>(np.pi/2 - pm) and angle<(np.pi/2 + pm)):
                    count2 = count2 + 1

    print("Fraction within +- 25 degrees: ", count2/count1)

    num_bins=100
    #plt.yscale('log')
    plt.hist(values, num_bins, density=True, alpha=0.5)
    plt.title(title)
    plt.xlim((0, np.pi))
    plt.savefig(path + "/" + idt + "_angle_distribution_" + p_label + ".png")
    plt.clf()    


def plot_distribution(values, per_values, per_labels, path, title, idt):

    for (p, l) in zip(per_values, per_labels):
        label = "%.1f" % l
        plt.axvline(x=p, label=label)

    num_bins=100
    plt.hist(values, num_bins, alpha=0.5, density=True)
    plt.title(title)
    plt.legend()
    plt.savefig(path + "/" + idt + "_distribution.png")
    plt.clf()


def plot_original_values(x_part, y_part, lognorm, path, idt, dimensions):

    max_val = max(np.max(x_part), np.max(y_part))
    
    if(lognorm):
        norm = cl.LogNorm(0.1, max_val + 0.1)
    else:
        norm = cl.Normalize(0.0, max_val)
    
    titles = ["X component", "Y component"]
    filename = path + "/" + idt + "_original_values.png"

    pl.plot_magnitude([x_part, y_part], 2*[norm], dimensions, \
            titles, filename)


def find_components(data, e_alpha, e_beta):

    # TODO implement in pp instead
    
    return [op.calc_magnitude(pp.calc_projection_vectors(data, \
            e, time_dependent=False), time_dependent=False) \
            for e in [e_alpha, e_beta]]



def calc_percentile_values(org, z_dir, per_value):
    
    X, Y = org.shape[:2]

    new = np.zeros((X, Y))

    for x in range(X):
        for y in range(Y):
            if(np.linalg.norm(org[x, y]) > per_value):
                new[x, y] = np.linalg.norm(z_dir[x, y])/\
                        (np.linalg.norm(org[x, y]))
    return new
    

def plot_percentile_values(org, x_frac, y_frac, percentiles, \
        per_values, lognorm, dimensions, titles, path, idt):

    for i in range(len(percentiles)):
        x_part, y_part = [calc_percentile_values(org, z, per_values[i]) \
                for z in [x_frac, y_frac]]

        max_val = max(np.max(x_part), np.max(y_part))

        if(lognorm):
            norm = cl.LogNorm(0.1, max_val + 0.1)
        else:
            norm = cl.Normalize(0.0, max_val)

        pl.plot_magnitude([x_part, y_part], 2*[norm], dimensions, \
            titles, path, idt + "_" + str(percentiles[i]))


def plot_values(values, e_alpha, e_beta, path, idt, \
        dimensions, title, lognorm):

    X, Y = disp.shape[:2]

    values_m = op.calc_magnitude(values, \
            over_time=False).reshape(X*Y)

    # plot distribution + percentiles
    fractions = np.linspace(0, 1, 11)[:3]
 
    # try fraction instead ...
    max_v = np.max(values_m)
    min_v = np.min(values_m)

    per_values = np.array([min_v + x*(max_v - min_v) for x in fractions])

    for i in range(len(fractions)):
        p_label = str(int(100*fractions[i]))
        plot_angle_distribution(values, -e_beta, per_values[i], p_label, path, \
            title, idt)

    plot_distribution(values_m, per_values, fractions, path, \
            title, idt)
     
    # first find x, y components
    x_values, y_values = find_components(values, e_alpha, e_beta)
    # plot original values
    
    plot_original_values(x_values, y_values, lognorm, path, \
            idt, dimensions)
    
    plot_angle_distribution(values, e_alpha, 0, '0', path, \
            title, idt)
    
    # plot perecentile 
    titles = [title + ", X fraction", \
              title + ", Y fraction"]


    # TODO: Rewrite to same formate as below

    plot_percentile_values(values, x_values, y_values, \
            fractions, per_values, lognorm, \
            dimensions, titles, path, idt)

    for i in range(len(fractions)):
        p_label = str(int(100*fractions[i]))
        plot_angle_distribution(values, e_alpha, per_values[i], p_label, path, \
            title, idt)
    

try:
    f_in = sys.argv[1]
except:
    print("Give file name as arguments.")
    exit(-1)

de = io.get_os_delimiter()
path = "Figures" + de + "Plot disp str"
io.make_dir_structure(path)

last_fn = f_in.split("/")[-1].split(".")
prefix, suffix = last_fn
dimensions = (664.30, 381.55)
 
disp, strain, e_alpha, e_beta = read_values(f_in, prefix, dimensions)

plot_values(disp, e_alpha, e_beta, path, prefix + "_disp", \
        dimensions, "Displacement", False)
plot_values(strain, e_alpha, e_beta, path, prefix + "_strain", \
        dimensions, "Principal strain", True)
