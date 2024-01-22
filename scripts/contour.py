# First test  for pyplot

import matplotlib.pyplot as plt
import os, csv
import numpy as np
import pandas as pd
from itertools import count
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from array import array
import seaborn as sns
from scipy.interpolate import griddata

dir = '../data/'
filename = 'Skynet_59735_AP_-_ST3_81159_29887.A.raw.txt'

def read_raw(path):
    outlist = []
    i = 0

    with open(path, 'r') as f:
        
        freader = csv.reader(f, delimiter='	')
        for row in freader:
            if i >= 69:
                l = []
                row = row[0].split()
                l.append(float(row[0]))
                l.append(float(row[1]))
                l.append(float(row[2]))
                l.append(float(row[5]))
                l.append(float(row[6]))
                l.append(float(row[7]))
                l.append(float(row[8]))
                l.append(float(row[9]))
                outlist.append(l)
                i += 1
            else:
                i += 1

    return outlist

def create_contour(data):
    # Extracting RA, DEC, and power data from columns
    #data.to_numpy()
    print(data)
    ra = data.iloc[:, 1]
    dec = data.iloc[:, 2]
    power0 = data.iloc[:, 3] # xx1

    # Creating a grid for the contour plot
    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()

    ra_grid, dec_grid = np.meshgrid(np.linspace(ra_min, ra_max,100), np.linspace(dec_min, dec_max, 100))

    # Interpolating power data to fill missing values
    method = 'cubic'
    cmap = 'mako'
    power_interp0 = griddata((ra, dec), power0, (ra_grid, dec_grid), method=method) #xx1

    # Plotting the contour plot
    levels = 5
    plt.contour(ra_grid, dec_grid, power_interp0, cmap='Grays', levels=levels, linestyles='dashed')
    plt.colorbar(label='Power')
    plt.gca().invert_xaxis()
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.title('Contour Plot - {} Levels in Grey'.format(levels))
    plt.show()

def create_contourf(data):
    # Extracting RA, DEC, and power data from columns
    #data.to_numpy()
    print(data)
    ra = data.iloc[:, 1]
    dec = data.iloc[:, 2]
    power0 = data.iloc[:, 3] # xx1

    # Creating a grid for the contour plot
    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()

    ra_grid, dec_grid = np.meshgrid(np.linspace(ra_min, ra_max,100), np.linspace(dec_min, dec_max, 100))

    # Interpolating power data to fill missing values
    method = 'cubic'
    cmap = 'mako'
    power_interp0 = griddata((ra, dec), power0, (ra_grid, dec_grid), method=method) #xx1

    # Plotting the contour plot
    plt.contourf(ra_grid, dec_grid, power_interp0, cmap=cmap)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.title('Contour Plot of Greenbank Radio Telescope Data')
    plt.colorbar(label='Power')
    plt.show()

def create_contour_combined(data):
    # Extracting RA, DEC, and power data from columns
    #data.to_numpy()
    print(data)
    ra = data.iloc[:, 1]
    dec = data.iloc[:, 2]
    power0 = data.iloc[:, 3] # xx1

    # Creating a grid for the contour plot
    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()

    ra_grid, dec_grid = np.meshgrid(np.linspace(ra_min, ra_max,100), np.linspace(dec_min, dec_max, 100))

    # Interpolating power data to fill missing values
    method = 'cubic'
    cmap = 'mako'
    power_interp0 = griddata((ra, dec), power0, (ra_grid, dec_grid), method=method) #xx1

    # Plotting the contour plot
    levels = 5
    contourf_plot = plt.contourf(ra_grid, dec_grid, power_interp0, cmap=cmap, levels=levels)
    contour_lines = plt.contour(ra_grid, dec_grid, power_interp0, colors='k', levels=levels, linestyles='dashed')
    
    # Invert X axis to match RA direction
    plt.gca().invert_xaxis()

    # Add color bar for filled contours, and a legend for contour lines
    plt.colorbar(contourf_plot, label='Power')
    plt.legend([contour_lines.collections], ['Contour Lines'])

    # Adding labels and title
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.title('Contour Plot - {} Levels Line and Fill'.format(levels))
    plt.show()

def create_contourf_all(data):
    # Extracting RA, DEC, and power data from columns
    #data.to_numpy()
    print(data)
    ra = data.iloc[:, 1]
    dec = data.iloc[:, 2]
    power0 = data.iloc[:, 3] # xx1
    power1 = data.iloc[:, 4] # yy1
    power2 = data.iloc[:, 5] # xx2
    power3 = data.iloc[:, 6] # yy2

    # Creating a grid for the contour plot
    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()

    ra_grid, dec_grid = np.meshgrid(np.linspace(ra_min, ra_max,100), np.linspace(dec_min, dec_max, 100))

    # Interpolating power data to fill missing values
    method = 'cubic'
    power_interp0 = griddata((ra, dec), power0, (ra_grid, dec_grid), method=method) #xx1
    power_interp1 = griddata((ra, dec), power1, (ra_grid, dec_grid), method=method) #yy1
    power_interp2 = griddata((ra, dec), power2, (ra_grid, dec_grid), method=method) #xx2
    power_interp3 = griddata((ra, dec), power3, (ra_grid, dec_grid), method=method) #yy2

    # Plotting the contour plot
    fig, axs = plt.subplots(nrows = 2, ncols = 2, gridspec_kw={'hspace': 0.3})
    axs = axs.flatten()
    
    all_max = max(power_interp0.max(), power_interp0.max(), power_interp1.max(), power_interp3.max())
    levels = np.linspace(0, max(power_interp0.max(), power_interp0.max(), power_interp1.max(), power_interp3.max()),100)
    cmap = 'mako'

    axs[0].contourf(ra_grid, dec_grid, power_interp0, cmap=cmap, levels = levels, vmin=power_interp0.min(), vmax=power_interp0.max())
    axs[0].set_title('XX1')
    axs[0].invert_xaxis()
    #axs[0].set_label
    im0 = axs[0].imshow(power_interp0)

    axs[2].contourf(ra_grid, dec_grid, power_interp2, cmap=cmap, levels = levels, vmin=0, vmax=all_max)
    axs[2].set_title('XX2')
    axs[2].invert_xaxis()
    im2 = axs[2].imshow(power_interp2)

    axs[1].contourf(ra_grid, dec_grid, power_interp1, cmap=cmap, levels = levels, vmin=0, vmax=all_max)
    axs[1].set_title('YY1')
    axs[1].invert_xaxis()
    im1 = axs[1].imshow(power_interp1)

    axs[3].contourf(ra_grid, dec_grid, power_interp3, cmap=cmap, levels = levels, vmin=0, vmax=all_max)
    axs[3].set_title('YY2')
    axs[3].invert_xaxis()
    im3 = axs[3].imshow(power_interp3)

    for ax in axs.flat:
        ax.set(xlabel='RA', ylabel='DEC')
    
    fig.suptitle('Contour Plot of Greenbank Radio Telescope Data', y = 0.95)
    #fig.tight_layout()

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im3, cax=cbar_ax, label='Power')
    #fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, label='Power')


    plt.show()

def main():
    path = str(dir+filename)
    plot_list = read_raw(path)

    out_list = []

    for line in plot_list:
        # Create the list with all data from file
        # 0 = time, 1 = RA, 2 = dec, 3 = xx1, 4 = yy1, 5 = xx2, 6 = yy2, 7 = calibration
        if abs(line[7]) != 1:
            l = []
            l.append(line[0])
            l.append(float(line[1]))
            l.append(float(line[2]))
            l.append(float(line[3]))
            l.append(float(line[4]))
            l.append(float(line[5]))
            l.append(float(line[6]))
            l.append(line[7])
            out_list.append(l)
        else:
            pass
    
    # Convert list to dataframe
    df = pd.DataFrame(out_list, columns=["time", "ra", "dec", "xx1", "yy1", "xx2", "yy2", "cal"])

    # create_scatter(df)
    create_contour_combined(df)
    # create_displot(df)
    # create_histplot(df)
    #create_kde(df)
    # create_normplot(df)

if __name__ == '__main__':
    main()
