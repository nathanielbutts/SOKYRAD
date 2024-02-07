from astropy.io import fits

dir = '../data/'
filename = 'Skynet_59735_AP_-_ST3_81159_29887.A.cal_XX0_img.fits'
path = dir + filename

with  fits.open(path) as hdul:
    hdul.verify('fix')
    hdr = hdul[0].header
    # print(repr(hdr))
    data = hdul[0].data
    mask = data > 0
    print(len(data[mask]))

#print(data[0])