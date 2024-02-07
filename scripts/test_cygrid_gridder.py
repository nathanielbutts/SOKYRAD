### TO DO:
# -[x] Add more colormap options
# -[ ] Figure out why RA displays in hr/min, not degree
# -[ ] Check dependencies and clean up if needed
# -[ ] Clean code into more defs
# -[ ] Work on integrating URL downloads

### List of functions
#  read_raw(path)
#      goes to the file path, then iterates through that file and puts all data into a single dataframe
#  setup_data(df, chan)
#      retrieves RA, DEC, and signal from the called channel from the dataframe from read_raw
#  setup_header(mapcenter, mapsize, beamsize_fwhm)
#      uses the calculated or manual mapcenter, mapsize, and beamsize_fwhm to make a FITS compatible
#      header.  This is necessary for the WCS routine and datacube in cygrid
#  create_map_and_grid(dec, ra, beamsize_fwhm, xcoords, ycoords, signal)
#      Calculates beamsize_fwhm, mapcenter, mapsize from the data in xcoords, ycoords from setup_data
#      Also creates the kernel, grid, and datacube from gridder and datacube for cygrid
#############
# Need to create
#   plot_one()
#   plot_many()
#   setup_data_many()
#   Change setup_data to setup_data_one()
#  main()
#      Constructs the path and calls read_raw()
#      sets characteristics for channels to plot as signal
#      calls read_raw, setup_data, setup_header, create_map_and_grid
#      plots the data


import matplotlib.pyplot as plt
import os, csv, cygrid
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.interpolate import griddata

# Paths to read data files
dir = '../data/'
#filename = 'test_ezb_data.txt'
filename = 'Skynet_60333_barnard_33-nb_107271_56634.A.cal.txt'
#filename = 'Skynet_59735_AP_-_ST3_81159_29887.A.cal.txt'
# dataurl = 'https://www.gb.nrao.edu/20m/peak/AP_-_ST3/Skynet_59735_AP_-_ST3_81159_29887.A.raw.txt' #not used, yet

# Take text file from skynet and parse into a list for later processing
def read_raw(path): #this should be complete
    outlist = []
    i = 0

    with open(path, 'r') as f:
        
        freader = csv.reader(f)#, delimiter=' ')
        for row in freader:
            if i >= 69: #usually 69
                l = []
                row = row[0].split()
                l.append(float(row[0])) # time
                l.append(float(row[1])) # ra
                l.append(float(row[2])) # dec
                l.append(float(row[5])) # xx1
                l.append(float(row[6])) # yy1
                l.append(float(row[7])) # xx2
                l.append(float(row[8])) # yy2
                l.append(float(row[9])) # cal
                if int(row[9]) != 0:
                    i += 1
                else:
                    outlist.append(l)
                    i += 1
            else:
                i += 1

    df = pd.DataFrame(outlist, columns=["time", "ra", "dec", "xx1", "yy1", "xx2", "yy2", "cal"])
    print(df)
    return df

def setup_data(df, chan): # Need to add all columns, currently only three
    global ra, dec
    xcoords = np.array(df.iloc[:, 1])
    ycoords = np.array(df.iloc[:, 2])
    signal = np.array(df.iloc[:, chan])
    
    ra, dec = xcoords, ycoords

    return xcoords, ycoords, signal

# Creates a FITS header.  This is used in the datacube portion later
def setup_header(mapcenter, mapsize, beamsize_fwhm): #seems to work...
    '''
    Produce a FITS header that contains the target field.
    '''

    # define target grid (via fits header according to WCS convention)
    # a good pixel size is a third of the FWHM of the PSF (avoids aliasing)
    pixsize = beamsize_fwhm / 3 #was 4.5
    dnaxis1 = int(mapsize[0] / pixsize)
    dnaxis2 = int(mapsize[1] / pixsize)

    header = {
        'NAXIS': 2,
        'NAXIS1': dnaxis1,
        'NAXIS2': dnaxis2,
        'CTYPE1': 'RA---SIN',
        'CTYPE2': 'DEC--SIN',
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'CDELT1': -pixsize,
        'CDELT2': pixsize,
        'CRPIX1': dnaxis1 / 2.,
        'CRPIX2': dnaxis2 / 2.,
        'CRVAL1': mapcenter[0],
        'CRVAL2': mapcenter[1],
        }

    return header

def create_map_and_grid(dec, ra, beamsize_fwhm, xcoords, ycoords, signal):
    # Create map center (tuple)? for plotting
    mapcenter = xcoords.min()+(xcoords.max()-xcoords.min())/2, ycoords.min()+(ycoords.max()-ycoords.min())/2
    #mapcenter = ra, dec
    #mapsize = 30., 30.
    mapsize = xcoords.max()-xcoords.min(), ycoords.max()-ycoords.min() #Can set manually or automatically, dealers choice.
    
    target_header = setup_header(mapcenter, mapsize, beamsize_fwhm)

    # let's already define a WCS object for later use in our plots:
    target_wcs = WCS(target_header)
    
    gridder = cygrid.WcsGrid(target_header)
    
    # This is used by cygrid.  I played with the kernelsize_FWHM till it "looked ok"
    kernelsize_fwhm = beamsize_fwhm/2.5  # degrees
    # see https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    kernelsize_sigma = kernelsize_fwhm / np.sqrt(8 * np.log(2))
    sphere_radius = 6. * kernelsize_sigma #again, guesses

    gridder.set_kernel(
        'gauss1d', # says it's 1d, but actually 2d.  gauss2d is for eliptical plots
        (kernelsize_sigma,),
        sphere_radius,
        kernelsize_sigma / 3. # played with this until it looked ok
        )

    # gridder is kinda like scipy.interpolate.griddata
    gridder.grid(xcoords, ycoords, signal)
    # datacube puts it all together
    cygrid_map = gridder.get_datacube()
    
    # Create size, how far apart to set spacing, and center for grid
    grid_size = (1000000, 1000000)  # Number of grid points in each dimension
    grid_spacing = (0.001, 0.001)  # Grid spacing in radians.  No difference between 0.01 and 0.00001
    grid_center = (xcoords.min()+(xcoords.max()-xcoords.min()/2), ycoords.min()+(ycoords.max()-ycoords.min())/2)
    #grid_center = ra, dec
    return target_wcs, cygrid_map

def main():
    #imkw used in plotting later
    # interpolation options
    #'bilinear', 'bicubic', '2spline16', 1'spline36', 'quadric', 'gaussian','lanczos', 
     

    # put toghet strings to make then feed into plot function
    path = str(dir+filename)
    df = read_raw(path)
    #df = create_df(plot_list)

    # Get data for gridder
    chan = 3 #3 = xx1, 4 = yy1, 5 = xx2, 6 = yy2
    xcoords, ycoords, signal = setup_data(df, chan) #plot three 

    beamsize_fwhm = 0.734 #degrees
    target_dec, target_ra = 0, 0
    target_wcs, cygrid_map = create_map_and_grid(target_dec, target_ra, beamsize_fwhm, xcoords, ycoords, signal)
    
    # to do comparison charts I had 4 total, uncomment all ax2, im2, etc to show them.
    fig = plt.figure(figsize=(14, 11))
    ax1 = fig.add_subplot(111, projection=target_wcs.celestial, aspect='auto') #change 111 to 221 if using 4 plots
    # ax2 = fig.add_subplot(222, projection=target_wcs.celestial, aspect='auto')
    # ax3 = fig.add_subplot(223, projection=target_wcs.celestial, aspect='auto')
    # ax4 = fig.add_subplot(224, projection=target_wcs.celestial, aspect='auto')
    
    # levels and linestyles used to create contours, if desired
    levels = 10
    linestyle = 'solid' # None, 'solid', 'dashed', 'dashdot', 'dotted'


    # some preferred cmaps:
    # 'Accent', 'Accent_r', ''CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 
    # 'PuBu', 'PuBu_r', 'RdBu', 'RdBu_r', 'RdYlBu', 'RdYlBu_r', 'Spectral', 'Spectral_r', 'brg', 'bwr', 'cubehelix',  
    # 'gist_rainbow', 'gist_rainbow_r', 'gnuplot', 'gnuplot2', 'hot', 'jet', 'jet_r', 'magma', 'magma_r', 'plasma', 
    # 'plasma_r', 'seismic', 'seismic_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 
    # 'tab20c_r', 'turbo', 'turbo_r',  ''viridis', 'viridis_r'
    # Jet and turbo seem best

    imkw = dict(origin='lower', interpolation='spline36') # spline36
    im1 = ax1.imshow(cygrid_map, **imkw, cmap='jet') 
    # im2 = ax2.imshow(cygrid_map, **imkw, cmap='jet') 
    # im3 = ax3.imshow(cygrid_map, **imkw, cmap='turbo') 
    # im4 = ax4.imshow(cygrid_map, **imkw, cmap='jet') 

    # uncomment cs1 and cs2 to make contour maps
    # cs1 = ax1.contour(
    #     cygrid_map, colors='white',
    #     levels=levels, alpha=0.5
    #     )
    # cs2 = ax2.contour(
    #     cygrid_map, colors='white',
    #     levels=levels, alpha=0.5
    #     )
    
    # plt.clabel(cs1, inline=True, fontsize=8, fmt='%1.1f')  # Adjust fontsize and format as needed
    # plt.clabel(cs2, inline=True, fontsize=8, fmt='%1.1f')  # Adjust fontsize and format as needed

    # uncomment suptitle if you use multiple plots
    fig.suptitle('Barnard 33 - Greenbank 20m', y=0.95)  # Adjust the y position
    if chan == 3:
        chan_title = 'XX1'
    elif chan == 4:
        chan_title = 'YY1'
    elif chan == 5:
        chan_title = 'XX2'
    else:
        chan_title = 'YY2'

    ax1.set_title('{}'.format(chan_title))
    # ax2.set_title('jet cmap w/contour')
    # ax3.set_title('turbo cmap')
    # ax4.set_title('jet cmap')

    # uncomment if using multiple plots
    plt.colorbar(im1, ax=ax1, label='Power(K)')
    # plt.colorbar(im2, ax=ax2, label='Power(K)')
    # plt.colorbar(im3, ax=ax3, label='Power(K)')
    # plt.colorbar(im4, ax=ax4, label='Power(K)')

    # uncomment and fix indents to make multiple plots
    # for ax in ax1, ax2, ax3, ax4: #if multiple, remove :#
    lon, lat = ax1.coords # if multiple, change ax2 to ax
    
    lon.set_axislabel('R.A. [deg]')
    lat.set_axislabel('Dec [deg]')

    plt.show()

if __name__ == '__main__':
    main()