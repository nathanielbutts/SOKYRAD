import urllib.request
import os, sys

dataurl = 'https://www.gb.nrao.edu/20m/peak/CRAB/Skynet_60335_crab_108895_58585.htm'


def retrieve_file(path, lastpath, filelist):
    if not os.path.isdir('.tmp'):
        os.mkdir('.tmp')
        print('Created .tmp directory')
    else:
        print('.tmp directory exists')
    for filename in filelist:
        fullpath = path + filename
        print(fullpath)
        with urllib.request.urlopen(fullpath) as f: #maybe just use requests.get(url,stream=True)??  https://stackoverflow.com/questions/30229231/python-save-image-from-url
            print("Downloading: {}".format(fullpath))
            html = f.read().decode('utf-8')
            f = open(str('.tmp/' + lastpath + filename), 'w')
            f.write(html)

def parse_address(dataurl):
    last_occurrence_index = dataurl.rfind('/')
    second_to_last_occurrence_index = dataurl.rfind('/', 0, last_occurrence_index)

    mainpath = 'https://www.gb.nrao.edu/20m/peak'
    secondpath = dataurl[second_to_last_occurrence_index:-3]
    lastpath = dataurl[last_occurrence_index:-1]

    return mainpath, secondpath, lastpath

def main():
    mainpath, secondpath, lastpath = parse_address(dataurl)
    path = mainpath + secondpath
    # At this URL, after the last number digit and period in the url, you will have files with endings:
    #    A.raw.txt, A.cal.txt, A.spect.cal.txt, A.caldata, A.cal_XX0_img.fits, A.cal_XX1_img.fits,  
    #        A.cal_YY0_img.fits, A.cal_YY1_img.fits, cyb.txt, spect.cyb.txt
    # You will also find these image files:
    #    A.pow.png, A.path.png, A.spect.cal.png, A.cal_XX0_img.png, A.cal_XX1_img.png
    #        A.cal_YY0_img.png, A.cal_YY1_img.png, cyb.txt.png, spect.cyb.txt.png

    ### !!! Won't let me download FITS files.  Error in "html = f.read()" of retrieve_file function
    ### Also won't download image files.
    ### Need to diagnose this later.

    filelist = ['A.raw.txt',
        'A.cal.txt', 
        'A.spect.cal.txt', 
        'A.caldata', 
        #'A.cal_XX0_img.fits', 
        #'A.cal_XX1_img.fits',
        #'A.cal_YY0_img.fits', 
        #'A.cal_YY1_img.fits',
        'cyb.txt', 
        'spect.cyb.txt']

    retrieve_file(path, lastpath, filelist)
    
if __name__ == '__main__':
    main()