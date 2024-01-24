import pandas as pd
import requests, re

dataurl = 'https://www.gb.nrao.edu/20m/peak/AP_-_ST3/Skynet_59735_AP_-_ST3_81159_29887.A.cal.txt'
r = requests.get(dataurl)

l = re.split(r'   ', r.text)
print(l)
# contour_data = pd.read_csv(r.content, delimiter=' ')
# print(contour_data.head())