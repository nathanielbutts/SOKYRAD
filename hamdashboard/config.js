// CUT START
var disableSetup = false; // Manually set to true to disable setup page menu option
var topBarCenterText = "KQ4TIV - EM66tu";

// Grid layout desired
var layout_cols = 4;
var layout_rows = 4;

// Menu items
// Structure is as follows: HTML Color code, Option, target URL, scaling 1=Original Size, side (optional, nothing is Left, "R" is Right)
// The values are [color code, menu text, target link, scale factor, side],
// add new lines following the structure for extra menu options. The comma at the end is important!
var aURL = [
  [
    "#f3de21",
    "SATS",
    "satellite.js",
    null,
    "undefined"
  ],
  [
    "#2196f3",
    "HAM LIVE",
    "https://www.ham.live/views/dashboard",
    1,
    "undefined"
  ],
  [
    "#2196f3",
    "CONTEST",
    "https://www.contestcalendar.com/fivewkcal.html",
    1,
    "undefined"
  ],
  [
    "#2196f3",
    "LIGHTNING",
    "https://map.blitzortung.org/#3.87/36.5/-89.41",
    1,
    "undefined"
  ],
  [
    "#2196f3",
    "RADAR",
    "dark|https://weather.gc.ca/?layers=alert,radar&center=43.39961001,-78.53212031&zoom=6&alertTableFilterProv=ON",
    1,
    "undefined"
  ],
  [
    "#2196f3",
    "WEATHER",
    "https://openweathermap.org/weathermap?basemap=map&cities=true&layer=radar&lat=36.976524&lon=-86.456017&zoom=5",
    1,
    "undefined"
  ],
  [
    "#2196f3",
    "WINDS",
    "https://earth.nullschool.net/#current/wind/isobaric/1000hPa/grid=on/orthographic=-86.13,36.51,3000",
    1,
    "undefined"
  ],
  [
    "",
    "Kentucky HAMS Discord",
    "https://discord.com/channels/1488707477099905074/1488709454848594001",
    "1",
    "undefined"
  ]
];

// Feed items
// Structure is as follows: target URL
// The values are [target link]
var aRSS = [
  [
    "https://www.amsat.org/feed/",
    60
  ],
  [
    "https://daily.hamweekly.com/atom.xml",
    120
  ]
];

// Dashboard Tiles items
// Tile Structure is Title, Source URL
// To display a website on the tiles use "iframe|" keyword before the tile URL
// [Title, Source URL],
// the comma at the end is important!
var aIMG = [

// Grid 1,1
  [
    [
      "Radar CONUS",
      "LOCAL RADAR",
      "LIGHTNING"
    ],
    "https://radar.weather.gov/ridge/standard/CONUS-LARGE_loop.gif",
    "https://radar.weather.gov/ridge/standard/KOHX_loop.gif",
    "https://images.lightningmaps.org/blitzortung/america/index.php?animation=usa"
  ],

// Grid 1,2
  [
    [
      "SATELLITE CAN",
      "SATELLITE CGL"
    ],
    "https://cdn.star.nesdis.noaa.gov/GOES16/GLM/SECTOR/can/EXTENT3/GOES16-CAN-EXTENT3-1125x560.gif",
    "https://cdn.star.nesdis.noaa.gov/GOES16/GLM/SECTOR/cgl/EXTENT3/GOES16-CGL-EXTENT3-600x600.gif"
  ],

// Grid 1,3
  [
    [
      "WINDS",
      "0-12 HOURS",
      "12-24 HOURS"
    ],
    "iframe|https://earth.nullschool.net/#current/wind/isobaric/1000hPa/grid=on/orthographic=-86.13,36.51,3000",
    "https://www.weather.gov/images/lmk/WebWxBrief/Next12Hours.gif",
    "https://www.weather.gov/images/lmk/WebWxBrief/Next12to24Hours.gif"
  ],

// Grid 1,4
  [
    [
      "TODAY",
      "TOMORROW",
      "+2 DAYS"
    ],
    "https://www.wpc.ncep.noaa.gov/noaa/noaad1.gif?1779394653",
    "https://www.wpc.ncep.noaa.gov/noaa/noaad2.gif?1779394653",
    "https://www.wpc.ncep.noaa.gov/noaa/noaad3.gif?1779394653"
  ],

  

// Grid 2,1
  [
    "",
    "https://www.hamqsl.com/solar101vhf.php",
    "https://www.hamqsl.com/solar100sc.php",
    "https://www.hamqsl.com/solarpich.php",
    "https://services.swpc.noaa.gov/images/animations/suvi/primary/map/latest.png"
  ],

// Grid 2,2
  [
    [
      "STEREO EUVI 195",
      "STEREO EUVI 304",
      "STEREO CORONA 1",
      "STEREO CORONA 2"
    ],
    "https://stereo-ssc.nascom.nasa.gov/beacon/latest_256/ahead_euvi_195_latest.jpg",
    "https://stereo-ssc.nascom.nasa.gov/beacon/latest_256/ahead_euvi_304_latest.jpg",
    "https://stereo-ssc.nascom.nasa.gov/beacon/latest_256/ahead_cor1_latest.jpg",
    "https://stereo-ssc.nascom.nasa.gov/beacon/latest_256/ahead_cor2_latest.jpg"
  ],

// Grid 2,3
  [
    "Radio Propagation",
    "https://www.tvcomm.co.uk/g7izu/Autosave/ATL_HF10_AutoSave.JPG",
    "https://www.tvcomm.co.uk/g7izu/Autosave/NA_ES_AutoSave.JPG",
    "https://www.short-wave.info/php/transmitter-site-map.php?mobile=false&lat=52.67|-21.96|-15.53|-9.42|-17.76|-17.53|46.34|50.73|42.81|39.75|50.89|29.60|6.23|39.40|-15.53|43.51|46.34|-21.96|34.38|44.15|39.36|46.34|39.91|39.91|46.34|27.46|24.88|27.46|36.28|39.36|42.04|36.28|36.21|12.69|18.22|24.17|42.04|29.60|-15.73|-7.90|36.21|12.69|36.21|29.15|30.65|-21.96|33.50|-38.83|36.28|36.21|27.46&lon=9.75|27.60|28.00|160.05|168.36|146.05|-67.83|4.39|23.19|116.81|-113.85|55.79|-10.70|32.86|28.00|-79.63|-67.83|27.64|108.61|86.90|75.72|-67.83|-76.58|-76.58|-67.83|-80.93|102.50|-80.93|-86.10|75.72|12.32|-86.10|-86.89|-8.02|-63.02|54.25|12.32|55.79|46.45|-14.38|-86.89|-8.02|-86.89|47.77|-87.09|27.64|-86.47|176.42|-86.10|-86.89|-80.93&freq=3975|4930|4965|5020|5040|5055|5130|5780|5900|5985|6030|6040|6050|6050|6065|6070|6160|6195|7285|7295|7415|7490|9265|9265|9330|9395|9440|9455|9475|9600|9705|9930|9980|11640|11775|11810|11870|11880|11965|12095|12160|13630|13845|15540|15555|15580|15610|15720|15810|15825|17790&az=ND|20|ND|ND|ND|ND|245|ND|126|257|ND|313|ND|310|315|ND|245|350|317|270|308|245|242|242|245|355|283|285|50|308|206|180|90|111|320|90|210|211|295|27|85|111|90|310|5|350|85|35|40|46|160"
  ],

// Grid 2,4
  [
    " ",
    "https://img.propagation.dr2w.de/n-america/10M/dr2w_animation_10M.gif",
    "https://img.propagation.dr2w.de/n-america/20M/dr2w_animation_20M.gif",
    "https://img.propagation.dr2w.de/n-america/40M/dr2w_animation_40M.gif",
    "https://www.sws.bom.gov.au/Images/HF%20Systems/Global%20HF/T%20Index%20Map/West/tindex.png"
  ],

// Grid 3,1
  [
    " ",
    "https://www.hamqsl.com/solarmuf.php",
    "https://www.hamqsl.com/solarmap.php",
    "https://services.swpc.noaa.gov/images/swx-overview-large.gif",
    "https://services.swpc.noaa.gov/images/animations/wam-ipe/wfs_ionosphere_new/latest.png"
  ],

// Grid 3,2
  [
    [
      "MUF",
      "FOF"
    ],
    "iframe|https://prop.kc2g.com/",
    "iframe|https://prop.kc2g.com/fof2/"
  ],

// Grid 3,3
  [
    "SPACE WEATHER",
    "https://services.swpc.noaa.gov/images/animations/ovation/north/latest.jpg",
    "https://services.swpc.noaa.gov/images/animations/ovation/south/latest.jpg",
    "https://services.swpc.noaa.gov/images/animations/enlil/latest.jpg",
    "https://services.swpc.noaa.gov/images/animations/geoelectric/InterMagEarthScope/EmapGraphics_1m/latest.png"
  ],

// Grid 3,4
  [
    [
      "ISS POSITION",
      "SO-59",
      "AO-91",
      "PO-101"
    ],
    "https://heavens-above.com/orbitdisplay.aspx?icon=iss&width=600&height=300&mode=M&satid=25544",
    "https://heavens-above.com/orbitdisplay.aspx?icon=default&width=600&height=300&mode=M&satid=27607",
    "https://heavens-above.com/orbitdisplay.aspx?icon=default&width=600&height=300&mode=M&satid=43017",
    "https://heavens-above.com/orbitdisplay.aspx?icon=default&width=600&height=300&mode=M&satid=43678"
  ],

// Grid 4,1
  [
    [
      "I65@22",
      "I65@28",
      "I65@32",
      "BG_WEST",
      "231@I165"
    ],
    "https://www.trimarc.org/images/milestone/CCTV_03_65_0223.jpg",
    "https://www.trimarc.org/images/milestone/CCTV_03_KY446_0002.jpg",
    "https://www.trimarc.org/images/milestone/CCTV_03_65_0327.jpg",
    "https://www.trimarc.org/images/milestone/CCTV_03_MorgantownRd_and_VeteransMemorialLn.jpg",
    "https://www.trimarc.org/images/milestone/CCTV_03_US231_0053.jpg"
  ],

// Grid 4,2
  [
    [
      "Scottsville Rd Central",
      "Scottsville Rd East",
      "Aviation Park",
      ""
    ],
    "https://webpubcontent.gray.tv/wbko/Weather/BGAirport.JPG",
    "https://webpubcontent.gray.tv/wbko/Weather/SleepInn.jpg",
    "https://webpubcontent.gray.tv/wbko/Weather/AviationPark.jpg"
  ],

// Grid 4,3
  [
    "Traffic Cam",
    "blob:https://smartway.tn.gov/48523bee-41de-48de-9cba-ceba435478bd"
  ],

// Grid 4,4
  [
    "TRIAL SQUARE",
    "https://www.weather.gov/images/lmk/wxstory/Tab2FileL.png",
    "https://www.weather.gov/images/lmk/wxstory/Tab3FileL.png",
    "https://weather.gov/images/lmk/WebWxBrief/Next12Hours.gif",
    "https://www.weather.gov/images/lmk/WebWxBrief/Next12to24Hours.gif"
  ],

];

// Image rotation intervals in milliseconds per tile - If the line below is commented, tiles will be rotated every 5000 milliseconds (5s)
var tileDelay = [
  10100,  // Grid 1,1
  10100,  // Grid 1,2
  10100,  // Grid 1,3
  10100,  // Grid 1,4
  10100,  // Grid 2,1
  10100,  // Grid 2,2
  10100,  // Grid 2,3
  10100,  // Grid 2,4
  10100,  // Grid 3,1
  10100,  // Grid 3,2
  10100,  // Grid 3,3
  10100,  // Grid 3,4
  5000,  // Grid 4,1
  5000,  // Grid 4,2
  5000,  // Grid 4,3
  5000  // Grid 4,4
];

// CUT END