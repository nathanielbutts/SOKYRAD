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
from scipy import interpolate

dir = '../data/'
filename = 'Skynet_60333_barnard_33-nb_107271_56634.A.cal.txt'
dataurl = 'https://www.gb.nrao.edu/20m/peak/AP_-_ST3/Skynet_59735_AP_-_ST3_81159_29887.A.cal.txt'

# Data starts at line 68
# Column definitions
#    0  UTC Time in S
#    1  Ra(deg)
#    2  Dec(deg)
#    3  Az(deg)
#    4  El(deg)
#    5  XX1
#    6  YY1
#    7  XX2
#    8  YY2
#    9  Calibration flag
#    10 Sweeps 1
#    11 Sweeps 2

def count_lines_file(path):
    with open(path, 'r') as f:
        return sum(1 for _ in f)

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
    # create pivot-table of data
    ra_grid, dec_grid, ra, dec, Z = create_xyz(data)

    # Create plot
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata
    #X, Y = np.meshgrid(X_unique, Y_unique)
    #length_values = len(X_unique) * len(Y_unique) #https://stackoverflow.com/questions/58177283/use-of-scipy-interpolate-griddata-for-interpolation-of-data-of-multiple-dimensio
    #points = np.empty((length_values, 2))
    #power_interp = griddata((ra, dec), power, (ra_grid, dec_grid), method='linear')
    # points[:, 0] = X.flatten()
    # points[:, 1] = Y.flatten()
    # print(points)
    f = interpolate.griddata((ra, dec), Z, (ra_grid, dec_grid), method='linear')
    fig, ax = plt.subplots()
    ax.contour(X, Y, f)

    # Invert xaxis to match SKYNET output
    ax.invert_xaxis()

    # Display the plot
    plt.show()
    
def create_scatter(data):
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x="time", y="xx1")
    plt.xlabel('time')
    plt.ylabel('xx1')
    plt.show()

def create_kde(data):
    #ax = sns.kdeplot(data, x="ra", y="dec", levels = 5, cmap="mako", fill=True, cumulative=True)
    # Z = pd.pivot_table(data, values="xx1", index="dec", columns="ra", fill_value=10)
    # X_unique = np.sort(data.ra.unique())
    # Y_unique = np.sort(data.dec.unique())

    X_unique, Y_unique, Z = create_xyz(data)
    X, Y = np.meshgrid(X_unique, Y_unique)

    fig, ax = plt.subplots()
    ax.contourf(X, Y, Z)
    #ax.invert_xaxis()
    plt.show()

def create_displot(data):
    sns.displot(data, x="ra", y="dec", kind="kde")
    plt.show()

def create_histplot(data):
    ax = sns.histplot(data, x="ra", y="xx1", bins = 1000, cmap = "mako")
    plt.show()

def create_normplot(data):
    fig, ax = plt.subplots()
    cmap = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])
    pcm = ax.pcolormesh(data, cmap=cmap, rasterized=True)
    fig.colorbar(pcm,ax=ax)
    plt.show()

def create_xyz(data):
    # original
    # https://alexmiller.phd/posts/contour-plots-in-python-matplotlib-x-y-z/
    # 1/19/24 19:11 creates plot, but is empty.  Think it's pulling only zero values.
    #Z = pd.pivot_table(data, values="xx1", columns="ra", index="dec", fill_value = 300) # original
    Z = data.pivot_table(values="xx1", columns="ra", index="dec", fill_value = 0).T.values # original
    X_unique = np.sort(data.ra.unique())
    Y_unique = np.sort(data.dec.unique())
    ra_min, ra_max = data.ra.min(), data.ra.max()
    dec_min, dec_max = data.dec.min(), data.dec.max()
    ra_grid, dec_grid = np.meshgrid(np.linspace(ra_min, ra_max, 100), np.linspace(dec_min, dec_max, 100))
    #ra, dec = data.ra.values(), data.dec.values()
    ra, dec = X_unique, Y_unique
    # f = open('output.txt', 'w')
    # for each in Z.values:
    #     f.write(str(each) + '\n')
    #Z = Z.flatten()
    return ra_grid, dec_grid, ra, dec, Z

# def create_xyz(data):
#     X_unique = np.sort(data.ra.unique())
#     Y_unique = np.sort(data.dec.unique())
#     # values = np.array(rain_at_locations.values.flatten())
#     Z = np.array(data.xx1.values.all()) # did have values.flatten()
#     return X_unique, Y_unique, Z

def main():
    path = str(dir+filename)
    #plot_list = read_raw(path)
    contour_data = pd.read_csv(path)
    print(contour_data.head())

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
    #df = pd.DataFrame(out_list, columns=["time", "ra", "dec", "xx1", "yy1", "xx2", "yy2", "cal"])

    # create_scatter(df)
    create_contour(df)
    # create_displot(df)
    # create_histplot(df)
    # create_kde(df)
    # create_normplot(df)

if __name__ == '__main__':
    main()

