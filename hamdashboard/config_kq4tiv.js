const topBarCenterText = `FFX ARES`;
// Menu items
// Structure is as follows HTML Color code, Option, target URL, scaling 1=Original Size, side (optional, nothing is Left, "R" is Right)
// The values are [color code, menu text, target link, scale factor, side],
// add new lines following the structure for extra menu options. The comma at the end is important!
const aURL = [
  ["add10d", "BACK", "#", "1"],
  ["add10d", "BACK", "#", "1", "R"],
  ["ff9100", "Refresh", "#", "1"],
  ["0dd1a7", "Help", "#", "1", "R"],
  [
    "FF000F",
    "WEA Alerts",
    "https://warn.pbs.org/",
    "1",
  ],
  ["FF000F", "WX Alerts", "https://alerts.weather.gov/search?zone=VAC059", "1", "R"],
  [
    "2196F3",
    "LIGHTNING",
    "https://map.blitzortung.org/#3.87/36.5/-89.41",
    "1",
    "R",
  ],
  [
    "FF000F", 
    "NWS SAFER Hazard", 
    "https://www.arcgis.com/apps/MapSeries/index.html?appid=ea8b0eeb2e9c45b790329c0ed2fdc225", 
    "1"
  ],
  [
    "2196F3", 
    "APRS", 
    "https://aprs.to/?center=38.8372,-77.1374&zoom=11", 
    "1"
  ],
  [
    "2196F3", 
    "WinlinkWed", 
    "https://www.qsl.net/kw4shp/WinlinkWed/WWmap.html", 
    "1",
  ],
  ["2196F3", "WINLINK", "https://cms.winlink.org:444/maps/propagation.aspx", "1"],
  ["2196F3", "Stuff In Space", "https://stuffin.space/", "1","R"],
  ["2196F3", "DX CLUSTER", "https://dxcluster.ha8tks.hu/map/", "1"],
  [
    "2196F3", 
    "SkyWarn", 
    "https://www.wx4lwx.org/index.php", 
    "1",
  ],
  [
    "2196F3",
    "RADAR",
    "https://radar.weather.gov/?settings=v1_eyJhZ2VuZGEiOnsiaWQiOiJ3ZWF0aGVyIiwiY2VudGVyIjpbLTc4LjIwMSwzOC42MTZdLCJsb2NhdGlvbiI6Wy03Ny4xMTQsMzguNzJdLCJ6b29tIjo3fSwiYW5pbWF0aW5nIjpmYWxzZSwiYmFzZSI6InN0YW5kYXJkIiwiYXJ0Y2MiOmZhbHNlLCJjb3VudHkiOmZhbHNlLCJjd2EiOmZhbHNlLCJyZmMiOmZhbHNlLCJzdGF0ZSI6ZmFsc2UsIm1lbnUiOnRydWUsInNob3J0RnVzZWRPbmx5IjpmYWxzZSwib3BhY2l0eSI6eyJhbGVydHMiOjAuNiwibG9jYWwiOjAuNiwibG9jYWxTdGF0aW9ucyI6MC44LCJuYXRpb25hbCI6MC42fX0%3D#/",
    "1",
    "R"
  ],
  [
    "2196F3",
    "WEATHER",
    "https://openweathermap.org/weathermap?basemap=map&cities=true&layer=temperature&lat=38.8474&lon=-77.3757&zoom=5",
    "1",
    "R",
  ],
  [
    "2196F3",
    "Surface Analysis",
    "https://www.wpc.ncep.noaa.gov/html/sfc-zoom.php",
    "1",
    "R",
  ],
  [
    "2196F3",
    "WINDS",
    "https://www.ventusky.com/?p=38.79;-77.16;8&l=wind-10m",
    "1",
    "R",
  ],
  [
    "2196F3",
    "Power Out",
    "https://ncrgdx.maps.arcgis.com/apps/dashboards/aa782813789c41a3b0b8fee04f01b2e6",
    "1",
  ],
  ["2196F3", "POTA Spots", "https://pota.app/#/", "1",],
  ["2196F3", "DX Heat", "https://dxheat.com/dxc/", "1",],
  [
    "2196F3",
    "Solar Ham",
    "https://solarham.com/",
    "1",
  ],	
  ["2196F3", "TIME.IS", "https://time.is/", "1",],
  [
    "2196F3",
    "Air Quality",
    "https://gispub.epa.gov/airnow/?forecastcontours=forecasttomorrow&tab=loops&xmin=-8868923.932654787&xmax=-8347011.903523855&ymin=4578979.162561644&ymax=4779855.67289499",
    "1",
    "R",
  ],
  ["2196F3", "NOAA HRRR-Smoke", "https://apps.gsl.noaa.gov/smoke/", "1","R"],
  [
    "2196F3",
    "Traffic",
    "https://trafficview.org/live_traffic/#10/38.8396/-77.2728",
    "1",
    "R",
  ],
  [
    "2196F3",
    "ADS-B",
    "https://globe.adsbexchange.com/",
    "1",
    "R",
  ],

];

// Dashboard items
// Structure is Title, Image Source URL
// [Title, Image Source URL],
// the comma at the end is important!
// You can't add more items because there are only 12 placeholders on the dashboard
// but you can replace the titles and the images with anything you want.
const aIMG = [
  [
    "RADAR", 
    "https://radar.weather.gov/ridge/standard/CONUS-LARGE_loop.gif",
    "https://radar.weather.gov/ridge/standard/KLWX_loop.gif",
  ],
  [
    "SATELLITE",
    "https://cdn.star.nesdis.noaa.gov/GOES16/GLM/CONUS/EXTENT3/GOES16-CONUS-EXTENT3-625x375.gif",
    "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/ne/GEOCOLOR/GOES16-NE-GEOCOLOR-600x600.gif",
  ],
  [
    "LIGHTNING",
    "https://images.lightningmaps.org/blitzortung/america/index.php?animation=usa",
    "https://www.blitzortung.org/en/Images/image_b_ny.png",
  ],
  [
    " ",
    "https://services.swpc.noaa.gov/images/animations/ovation/north/latest.jpg",
    "https://services.swpc.noaa.gov/images/animations/ovation/south/latest.jpg",
    "https://services.swpc.noaa.gov/images/animations/enlil/latest.jpg",
    "https://services.swpc.noaa.gov/images/animations/geoelectric/InterMagEarthScope/EmapGraphics_1m/latest.png",
  ],
  [
    "",
    "https://www.weather.gov/images/lwx/weatherstory.gif",
    "https://graphical.weather.gov/GraphicalNDFD.php?width=515&timezone=EDT&sector=CONUS&element=t&n=4",
    "https://www.wpc.ncep.noaa.gov/heat_index_MAX/bchi_day3.gif",
    "https://www.cpc.ncep.noaa.gov/products/stratosphere/uv_index/uvi_map.gif",
  ],
  [
     "Forecast & Activity",
	"https://www.wpc.ncep.noaa.gov/noaa/noaa.gif",
	"https://www.spc.noaa.gov/exper/mesoanalysis/activity_loop.gif",
	"https://www.spc.noaa.gov/products/watch/validww.png",
	"https://www.spc.noaa.gov/products/exper/day4-8/day48prob.gif",
	"https://www.wpc.ncep.noaa.gov/threats/final/hazards_d3_7_contours.png",
	"https://www.wpc.ncep.noaa.gov/qpf/fill_94qwbg.gif",
	"https://forecast.weather.gov/wwamap/png/lwx.png",
    "https://www.wpc.ncep.noaa.gov/sfc/namussfc12wbg.gif",
    "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/DMW/GOES16-ABI-CONUS-DMW.gif",
    "https://www.wpc.ncep.noaa.gov/medr/9jhwbg_conus.gif",
    "https://www.cocorahs.org/Maps/GetMap.aspx?state=VA&county=FX&type=precip",
    "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/10/latest.jpg",

  ],
  [
    " ",
    "https://www.hamqsl.com/solarmuf.php",
    "https://www.hamqsl.com/solarmap.php",
    "https://services.swpc.noaa.gov/images/swx-overview-large.gif",
    "https://services.swpc.noaa.gov/images/animations/wam-ipe/wfs_ionosphere_new/latest.png",

  ],
  [
    " ",
    "https://img.propagation.dr2w.de/n-america/10M/dr2w_animation_10M.gif",
    "https://img.propagation.dr2w.de/n-america/20M/dr2w_animation_20M.gif",
    "https://img.propagation.dr2w.de/n-america/40M/dr2w_animation_40M.gif",
    "https://services.swpc.noaa.gov/images/animations/d-rap/global/d-rap/latest.png",

  ],
  [
    "Snow & Ice Outlook",
    "https://www.weather.gov/images/lwx/winter/StormTotalSnowWeb1.jpg",
    "https://www.weather.gov/images/lwx/winter/StormTotalIceWeb1.jpg",	  
    "https://www.weather.gov/images/lwx/winter/SnowAmt90Prcntl.jpg",
    "https://www.weather.gov/images/lwx/winter/SnowAmt10Prcntl.jpg",	
    "https://www.weather.gov/images/lwx/winter/outlook/D3_WinterThreat.png",
    "https://www.weather.gov/images/lwx/winter/outlook/D4_WinterThreat.png",
    "https://www.weather.gov/images/lwx/winter/outlook/D5_WinterThreat.png",
    "https://www.weather.gov/images/lwx/winter/outlook/D6_WinterThreat.png",
    "https://www.weather.gov/images/lwx/winter/outlook/D7_WinterThreat.png",
    "https://www.weather.gov/images/lwx/winter/ProbSnowGETr.jpg",
    "https://www.weather.gov/images/lwx/winter/ProbSnowGE01.jpg",
  ],
  [
    "FFX Traffic",
    "https://cctv.trafficview.org/thumbnail/VDOT_179",
    "https://cctv.trafficview.org/thumbnail/VDOT_1025",
    "https://cctv.trafficview.org/thumbnail/VDOT_1025",
    "https://cctv.trafficview.org/thumbnail/VDOT_1435",
    "https://cctv.trafficview.org/thumbnail/VDOT_1450",
    "https://cctv.trafficview.org/thumbnail/VDOT_1713",
    "https://cctv.trafficview.org/thumbnail/VDOT_1710",
    "https://cctv.trafficview.org/thumbnail/VDOT_1454",
    "https://cctv.trafficview.org/thumbnail/VDOT_3660",
    "https://cctv.trafficview.org/thumbnail/VDOT_1708",
    "https://cctv.trafficview.org/thumbnail/VDOT_1432",
    "https://cctv.trafficview.org/thumbnail/VDOT_154",
    "https://cctv.trafficview.org/thumbnail/VDOT_1442",
  ],
  [
    "Radio Propagation",
    "https://www.tvcomm.co.uk/g7izu/Autosave/ATL_HF10_AutoSave.JPG",
    "https://www.tvcomm.co.uk/g7izu/Autosave/NA_ES_AutoSave.JPG",
    "https://www.short-wave.info/php/transmitter-site-map.php?mobile=false&lat=52.67|-21.96|-15.53|-9.42|-17.76|-17.53|46.34|50.73|42.81|39.75|50.89|29.60|6.23|39.40|-15.53|43.51|46.34|-21.96|34.38|44.15|39.36|46.34|39.91|39.91|46.34|27.46|24.88|27.46|36.28|39.36|42.04|36.28|36.21|12.69|18.22|24.17|42.04|29.60|-15.73|-7.90|36.21|12.69|36.21|29.15|30.65|-21.96|33.50|-38.83|36.28|36.21|27.46&lon=9.75|27.60|28.00|160.05|168.36|146.05|-67.83|4.39|23.19|116.81|-113.85|55.79|-10.70|32.86|28.00|-79.63|-67.83|27.64|108.61|86.90|75.72|-67.83|-76.58|-76.58|-67.83|-80.93|102.50|-80.93|-86.10|75.72|12.32|-86.10|-86.89|-8.02|-63.02|54.25|12.32|55.79|46.45|-14.38|-86.89|-8.02|-86.89|47.77|-87.09|27.64|-86.47|176.42|-86.10|-86.89|-80.93&freq=3975|4930|4965|5020|5040|5055|5130|5780|5900|5985|6030|6040|6050|6050|6065|6070|6160|6195|7285|7295|7415|7490|9265|9265|9330|9395|9440|9455|9475|9600|9705|9930|9980|11640|11775|11810|11870|11880|11965|12095|12160|13630|13845|15540|15555|15580|15610|15720|15810|15825|17790&az=ND|20|ND|ND|ND|ND|245|ND|126|257|ND|313|ND|310|315|ND|245|350|317|270|308|245|242|242|245|355|283|285|50|308|206|180|90|111|320|90|210|211|295|27|85|111|90|310|5|350|85|35|40|46|160",
  ],
  [
    "",
    "https://www.hamqsl.com/solar101vhf.php",
    "https://www.hamqsl.com/solar100sc.php",
    "https://www.hamqsl.com/solarpich.php",
    "https://services.swpc.noaa.gov/images/animations/suvi/primary/map/latest.png",
  ],
];

