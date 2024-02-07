import matplotlib.pyplot as plt
import os, csv
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator, InterpolatedUnivariateSpline, interp2d

dir = '../data/'
filename = 'Skynet_60335_crab_108895_58585.htA.raw.txt'

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

def plot(data):
    ra = data.iloc[:, 1]
    dec = data.iloc[:, 2]
    z = data.iloc[:, 3]

    xnew = np.arange(ra.min(), ra.max(), 100)
    ynew = np.arange(dec.min(), dec.max(),100)
    znew = interp2d(ra, dec, z, kind='cubic')
    znew = np.arange(z.min(), z.max(), 100)

    fig, ax = plt.subplots()

    ax.plot(xnew, ynew, 'o', markersize=2, color='lightgrey')
    #ax.tricontour(xnew, ynew, znew, levels=10)
    #ax.tripcolor(ra, dec, z)
    ax.scatter(ra, dec, s=z)
    plt.title('Ungridded Data')
    plt.show()
    return znew


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

    plot(df)
    

if __name__ == '__main__':
    main()
