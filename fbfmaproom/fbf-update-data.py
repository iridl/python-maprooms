import argparse
import cftime
import xarray as xr
import os
import shutil

import pingrid

base = "http://iridl.ldeo.columbia.edu"

url_datasets = [
    (
        "enso",
        "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version5/.sst/zlev/removeGRID/X/-170/-120/RANGE/Y/-5/5/RANGEEDGES/dup/T/12.0/splitstreamgrid/dup/T2/(1856)/last/RANGE/T2/30.0/12.0/mul/runningAverage/T2/12.0/5.0/mul/STEP/%5BT2%5DregridLB/nip/T2/12/pad1/T/unsplitstreamgrid/sub/%7BY/cosd%7D%5BX/Y%5Dweighted-average/T/3/1.0/runningAverage/%7BLaNina/-0.45/Neutral/0.45/ElNino%7Dclassify/T/-2/1/2/shiftdata/%5BT_lag%5Dsum/5/flagge/T/-2/1/2/shiftdata/%5BT_lag%5Dsum/1.0/flagge/dup/a%3A/sst/(LaNina)/VALUE/%3Aa%3A/sst/(ElNino)/VALUE/%3Aa/add/1/maskge/dataflag/1/index/2/flagge/add/sst/(phil)/unitmatrix/sst_out/(Neutral)/VALUE/mul/exch/sst/(phil2)/unitmatrix/sst_out/(LaNina)/(ElNino)/VALUES/%5Bsst_out%5Dsum/mul/add/%5Bsst%5Ddominant_class//long_name/(ENSO%20Phase)/def/startcolormap/DATA/1/3/RANGE/blue/blue/blue/grey/red/red/endcolormap/T/(1980)/last/RANGE/"
    ),
    (
        "enso-4mo",
        "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version5/.sst/zlev/removeGRID/X/-170/-120/RANGE/Y/-5/5/RANGEEDGES/dup/T/12.0/splitstreamgrid/dup/T2/(1856)/last/RANGE/T2/30.0/12.0/mul/runningAverage/T2/12.0/5.0/mul/STEP/%5BT2%5DregridLB/nip/T2/12/pad1/T/unsplitstreamgrid/sub/%7BY/cosd%7D%5BX/Y%5Dweighted-average/T/4/1.0/runningAverage/%7BLaNina/-0.45/Neutral/0.45/ElNino%7Dclassify/T/-2/1/2/shiftdata/%5BT_lag%5Dsum/5/flagge/T/-2/1/2/shiftdata/%5BT_lag%5Dsum/1.0/flagge/dup/a%3A/sst/(LaNina)/VALUE/%3Aa%3A/sst/(ElNino)/VALUE/%3Aa/add/1/maskge/dataflag/1/index/2/flagge/add/sst/(phil)/unitmatrix/sst_out/(Neutral)/VALUE/mul/exch/sst/(phil2)/unitmatrix/sst_out/(LaNina)/(ElNino)/VALUES/%5Bsst_out%5Dsum/mul/add/%5Bsst%5Ddominant_class//long_name/(ENSO%20Phase)/def/startcolormap/DATA/1/3/RANGE/blue/blue/blue/grey/red/red/endcolormap/T/(1980)/last/RANGE/",
    ),
    (
        "rain-malawi",
        "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.Merged_Analysis/.monthly/.latest/.ver2/.prcp_est/X/32/36/RANGE/Y/-17/-9/RANGE/T/(Dec-Feb)/seasonalAverage/",
    ),
    (
        "pnep-malawi",
        "http://iridl.ldeo.columbia.edu/home/.remic/.IRI/.FD/.NMME_Seasonal_HFcast_Combined/.malawi/.nonexceed/.prob/",
    ),
    (
        "madagascar/enacts-precip-djf",
        "http://map.meteomadagascar.mg/SOURCES/.Madagascar_v4/.ALL/.monthly/.rainfall/.rfe/T/(Dec-Feb)/seasonalAverage//units/(mm/month)/def/X/42.525/48.975/RANGE/Y/-25.9875/-20.025/RANGE/",
    ),
    (
        "madagascar/enacts-precip-mon-djf",
        "http://map.meteomadagascar.mg/SOURCES/.Madagascar_v4/.MON/.monthly/.rainfall/.rfe/T/(Dec-Feb)/seasonalAverage//units/(mm/month)/def/X/42.525/48.975/RANGE/Y/-25.9875/-20.025/RANGE/",
    ),
    (
        "madagascar/chirps-precip-djf",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/42.525/48.975/RANGE/Y/-25.9875/-20.025/RANGE/T/(Dec-Feb)/seasonalAverage/c%3A/3//units//months/def/%3Ac/mul//name//precipitation/def/",
    ),
    (
        "madagascar/ndvi-djf",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.SAF/.NDVI/X/42.525/.0375/48.975/GRID/Y/-25.9875/.0375/-20.025/GRID/T/(Dec-Feb)/seasonalAverage/",
    ),
    (
        "madagascar/wrsi-djf",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.EROS/.FEWS/.dekadal/.SAF/.Maize/.do/X/42.525/48.975/RANGE/Y/-25.9875/-20.025/RANGE/T/(Dec-Feb)/seasonalAverage/T/(months%20since%201960-01-01)/streamgridunitconvert/",
    ),
    (
        "madagascar/enacts-mon-spi-djf",
        "http://map.meteomadagascar.mg/SOURCES/.Madagascar_v4/.MON/.seasonal/.rainfall/.SPI-3-month/.spi/T/(Dec-Feb)/VALUES/",
    ),
    (
        "madagascar/pnep-djf",
        "http://iridl.ldeo.columbia.edu/home/.aaron/.DGM/.Forecast/.Seasonal/.NextGen/.Madagascar_Full/.DJF/.NextGen/.FbF/.pne/S/(0000%201%20Sep)/(0000%201%20Oct)/(0000%201%20Nov)/VALUES/",
    ),
    (
        'madagascar/subseas-dry-tercile-snov-l22-djf',
        '{url}',
        (
            {'url': 'http://iridl.ldeo.columbia.edu/SOURCES/.IRI/.FD/.SubX/.SubX_Hindcast/.Subx_biweekly_Precipitation/.MME_v21_Precipitation_ELR/.prob/L/22/VALUE/L/removeGRID/S/(0000%206%20Nov%201999)/(0000%204%20Nov%202000)/(0000%203%20Nov%202001)/(0000%202%20Nov%202002)/(0000%201%20Nov%202003)/(0000%206%20Nov%202004)/(0000%205%20Nov%202005)/(0000%204%20Nov%202006)/(0000%203%20Nov%202007)/(0000%201%20Nov%202008)/(0000%207%20Nov%202009)/(0000%206%20Nov%202010)/(0000%205%20Nov%202011)/(0000%203%20Nov%202012)/(0000%202%20Nov%202013)/(0000%201%20Nov%202014)/(0000%207%20Nov%202015)/(0000%205%20Nov%202016)/VALUES/C/(Below_Normal)/VALUE/C/removeGRID/X/42.525/48.975/RANGEEDGES/Y/-25.9875/-20.025/RANGEEDGES/S/(months%20since%201960-01-01)/streamgridunitconvert/S/toi4/use_as_grid/S/2.5/shiftGRID/S//T/renameGRID/'},
            {'url': 'https://iridl.ldeo.columbia.edu/SOURCES/.IRI/.FD/.SubX/.SubX_Tercile_Forecast/.Subx_biweekly_Precipitation/.MMEv21_Precipitation_ELR/.prob/L/22/VALUE/L/removeGRID/C/(Below_Normal)/VALUE/C/removeGRID/S/(0000%206%20Nov%202020)/(0000%205%20Nov%202021)/(0000%204%20Nov%202022)/(0000%203%20Nov%202023)/VALUES/X/42.525/48.975/RANGEEDGES/Y/-25.9875/-20.025/RANGEEDGES/S/(months%20since%201960-01-01)/streamgridunitconvert/S/toi4/use_as_grid/S/2.5/shiftGRID/S//T/renameGRID/'}
        ),
    ),
    (
        "madagascar/obs-subseas-rainfall",
        "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.UNIFIED_PRCP/.GAUGE_BASED/.GLOBAL/.v1p0/.extREALTIME/.rain/T/14/runningAverage/c%3A/14.0//units//days/def/%3Ac/mul/T/(1200%2021%20Nov%201999%20-%201200%205%20Dec%201999)/(1200%2019%20Nov%202000%20-%201200%203%20Dec%202000)/(1200%2018%20Nov%202001%20-%201200%202%20Dec%202001)/(1200%2017%20Nov%202002%20-%201200%201%20Dec%202002)/(1200%2016%20Nov%202003%20-%201200%2030%20Nov%202003)/(1200%2021%20Nov%202004%20-%201200%205%20Dec%202004)/(1200%2020%20Nov%202005%20-%201200%204%20Dec%202005)/(1200%2019%20Nov%202006%20-%201200%203%20Dec%202006)/(1200%2018%20Nov%202007%20-%201200%202%20Dec%202007)/(1200%2016%20Nov%202008%20-%201200%2030%20Nov%202008)/(1200%2022%20Nov%202009%20-%201200%206%20Dec%202009)/(1200%2021%20Nov%202010%20-%201200%205%20Dec%202010)/(1200%2020%20Nov%202011%20-%201200%204%20Dec%202011)/(1200%2018%20Nov%202012%20-%201200%202%20Dec%202012)/(1200%2017%20Nov%202013%20-%201200%201%20Dec%202013)/(1200%2016%20Nov%202014%20-%201200%2030%20Nov%202014)/(1200%2022%20Nov%202015%20-%201200%206%20Dec%202015)/(1200%2020%20Nov%202016%20-%201200%204%20Dec%202016)/(1200%2019%20Nov%202017%20-%201200%203%20Dec%202017)/(1200%2018%20Nov%202018%20-%201200%202%20Dec%202018)/(1200%2017%20Nov%202019%20-%201200%201%20Dec%202019)/(1200%2022%20Nov%202020%20-%201200%206%20Dec%202020)/(1200%2021%20Nov%202021%20-%201200%205%20Dec%202021)/(1200%2020%20Nov%202022%20-%201200%204%20Dec%202022)/(1200%2019%20Nov%202023%20-%201200%203%20Dec%202023)/VALUES/T//pointwidth/0/def/pop/T/(months%20since%201960-01-01)/streamgridunitconvert/T/toi4/use_as_grid/T/2.5/shiftGRID/X/42.525/48.975/RANGEEDGES/Y/-25.9875/-20.025/RANGEEDGES/"
    ),
    (
        "madagascar/enacts-precip-ond",
        "http://map.meteomadagascar.mg/SOURCES/.Madagascar_v4/.ALL/.monthly/.rainfall/.rfe/T/(Oct-Dec)/seasonalAverage//units/(mm/month)/def/X/42.525/48.975/RANGE/Y/-25.9875/-20.025/RANGE/",
    ),
    (
        "madagascar/enacts-precip-mon-ond",
        "http://map.meteomadagascar.mg/SOURCES/.Madagascar_v4/.MON/.monthly/.rainfall/.rfe/T/(Oct-Dec)/seasonalAverage//units/(mm/month)/def/X/42.525/48.975/RANGE/Y/-25.9875/-20.025/RANGE/",
    ),
    (
        "madagascar/chirps-precip-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/42.525/48.975/RANGE/Y/-25.9875/-20.025/RANGE/T/(Oct-Dec)/seasonalAverage/c%3A/3//units//months/def/%3Ac/mul//name//precipitation/def/",
    ),
    (
        "madagascar/ndvi-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.SAF/.NDVI/X/42.525/.0375/48.975/GRID/Y/-25.9875/.0375/-20.025/GRID/T/(Oct-Dec)/seasonalAverage/",
    ),
    (
        "madagascar/wrsi-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.EROS/.FEWS/.dekadal/.SAF/.Maize/.do/X/42.525/48.975/RANGE/Y/-25.9875/-20.025/RANGE/T/(Oct-Dec)/seasonalAverage/T/(months%20since%201960-01-01)/streamgridunitconvert/",
    ),
    (
        "madagascar/enacts-mon-spi-ond",
        "http://map.meteomadagascar.mg/SOURCES/.Madagascar_v4/.MON/.seasonal/.rainfall/.SPI-3-month/.spi/T/(Oct-Dec)/VALUES/",
    ),
    (
        "madagascar/pnep-ond",
        "http://iridl.ldeo.columbia.edu/home/.aaron/.DGM/.Forecast/.Seasonal/.NextGen/.Madagascar_Full/.OND/.NextGen/.FbF/.pne/S/(1%20Jul)/(1%20Aug)/(1%20Sep)/VALUES/",
    ),
    (
        "madagascar/ndvi-viirs-ond",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/42.525/.25/48.975/GRID/Y/-25.9875/.25/-20.025/GRID/T/(Oct-Dec)/seasonalAverage/",
    ),
    (
        "madagascar/ndvi-viirs-djf",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/42.525/.25/48.975/GRID/Y/-25.9875/.25/-20.025/GRID/T/(Dec-Feb)/seasonalAverage/",
    ),
    (
        "ethiopia/rain-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/T/(Mar-May)/seasonalAverage/30/mul//units/(mm/month)/def/",
    ),
    (
        "ethiopia/rain-prev-seas-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/T/(Oct-Dec)/seasonalAverage/30/mul//units/(mm/month)/def/T/5/shiftGRID/",
    ),
    (
        "ethiopia/rain-prev-seas-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/T/(Mar-May)/seasonalAverage/30/mul//units/(mm/month)/def/T/7/shiftGRID/",
    ),
    (
        "ethiopia/spi-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/monthlyAverage/30/mul/a%3A/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7D/def/-1./shiftGRID/T/first/pointwidth/1/sub/add/last/RANGE//long_name/(Standardized%20Precipitation%20Index)/def/%3Aa%3A/T/3/runningAverage/T/12/splitstreamgrid/0./flaggt/%5BT2%5Daverage/1./3./div/flaggt/1./masklt/%5BT%5D/REORDER/CopyStream/%3Aa/mul/DATA/-3/3/RANGE//name//spi/def/T/(Mar-May%201981)/last/12/RANGESTEP/"
    ),
    (
        "ethiopia/spi-prev-seas-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/monthlyAverage/30/mul/a%3A/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7D/def/-1./shiftGRID/T/first/pointwidth/1/sub/add/last/RANGE//long_name/(Standardized%20Precipitation%20Index)/def/%3Aa%3A/T/3/runningAverage/T/12/splitstreamgrid/0./flaggt/%5BT2%5Daverage/1./3./div/flaggt/1./masklt/%5BT%5D/REORDER/CopyStream/%3Aa/mul/DATA/-3/3/RANGE//name//spi/def/T/(Mar-May%201981)/last/12/RANGESTEP/T/7/shiftGRID/",
    ),
    (
        "ethiopia/spi-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/monthlyAverage/30/mul/a%3A/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7D/def/-1./shiftGRID/T/first/pointwidth/1/sub/add/last/RANGE//long_name/(Standardized%20Precipitation%20Index)/def/%3Aa%3A/T/3/runningAverage/T/12/splitstreamgrid/0./flaggt/%5BT2%5Daverage/1./3./div/flaggt/1./masklt/%5BT%5D/REORDER/CopyStream/%3Aa/mul/DATA/-3/3/RANGE//name//spi/def/T/(Oct-Dec%201981)/last/12/RANGESTEP/",
    ),
    (
        "ethiopia/spi-prev-seas-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/monthlyAverage/30/mul/a%3A/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7D/def/-1./shiftGRID/T/first/pointwidth/1/sub/add/last/RANGE//long_name/(Standardized%20Precipitation%20Index)/def/%3Aa%3A/T/3/runningAverage/T/12/splitstreamgrid/0./flaggt/%5BT2%5Daverage/1./3./div/flaggt/1./masklt/%5BT%5D/REORDER/CopyStream/%3Aa/mul/DATA/-3/3/RANGE//name//spi/def/T/(Oct-Dec%201981)/last/12/RANGESTEP/T/5/shiftGRID/",
    ),
    (
	"ethiopia/rain-prev-partial-mam",
	"https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRP/.v1p0/.daily/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/T/(Oct-Nov)/seasonalAverage/30/mul//units/(mm/month)/def/T/5.5/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Mar-May)/seasonalAverage/",
    ),
    (
        "ethiopia/ndvi-nov-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/0.25/48.375/GRID/Y/2.625/0.25/15.375/GRID/T/%281-16%20Nov%29seasonalAverage/T/toi4/use_as_grid/T//pointwidth/0/def/5.5/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-dec-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Dec)/seasonalAverage/T/4/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-jan-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Jan)/seasonalAverage/T/3/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-feb-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Feb)/seasonalAverage/T/2/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-jun-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Jun)/seasonalAverage/T/5/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-jul-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Jul)/seasonalAverage/T/4/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-aug-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Aug)/seasonalAverage/T/3/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Oct-Dec)/seasonalAverage/",
    ),
    (
        "ethiopia/ndvi-prev-seas-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Mar-May)/seasonalAverage/T/7/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-prev-seas-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Oct-Dec)/seasonalAverage/T/5/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-viirs-mam",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Mar-May)/seasonalAverage/",
    ),
    (
        "ethiopia/ndvi-viirs-oct-mam",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Oct)/seasonalAverage/T/6/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-viirs-nov-mam",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/%281-16%20Nov%29seasonalAverage/T/toi4/use_as_grid/T//pointwidth/0/def/5.5/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-viirs-dec-mam",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Dec)/seasonalAverage/T/4/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-viirs-jan-mam",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Jan)/seasonalAverage/T/3/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-viirs-feb-mam",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Feb)/seasonalAverage/T/2/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-viirs-may-ond",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(May)/seasonalAverage/T/6/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-viirs-jun-ond",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Jun)/seasonalAverage/T/5/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-viirs-jul-ond",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Jul)/seasonalAverage/T/4/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-viirs-aug-ond",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Aug)/seasonalAverage/T/3/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-viirs-ond",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Oct-Dec)/seasonalAverage/",
    ),
    (
        "ethiopia/ndvi-viirs-prev-seas-ond",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Mar-May)/seasonalAverage/T/7/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-viirs-prev-seas-mam",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Oct-Dec)/seasonalAverage/T/5/shiftGRID/",
    ),
    (
        "ethiopia/pnep-mam",
        "http://iridl.ldeo.columbia.edu/home/.aaron/.Ethiopia/.CPT/.NextGen/.MAM_PRCP/.Ethiopia/.NextGen/.FbF/.pne/P//P//percentile/0/5/5/95/NewEvenGRID/replaceGRID/",
    ),
    (
        "ethiopia/rain-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/30/mul/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/T/(Oct-Dec)/seasonalAverage//units/(mm/month)/def/",
    ),
    (
        "ethiopia/pnep-ond",
        "http://iridl.ldeo.columbia.edu/home/.aaron/.Ethiopia/.CPT/.NextGen/.OND_PRCP/.Ethiopia/.NextGen/.FbF/.pne/P//P//percentile/0/5/5/95/NewEvenGRID/replaceGRID/"
    ),
    (
        'niger/enacts-precip-jas',
        'http://iridl.ldeo.columbia.edu/home/.aaron/.Niger/.ENACTS/.monthly/.rainfall/.CHIRPS/.rfe_merged/T/(1991)/last/RANGE/T/(Jul-Sep)/seasonalAverage/3/mul///name//obs/def/Y/first/17/RANGE/'
    ),
    (
        'niger/chirps-precip-jun',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/(0)/(16)/RANGE/Y/(11)/(17)/RANGE/T/(Jun)/seasonalAverage/T//pointwidth/0/def/2/shiftGRID/c%3A/1//units//months/def/%3Ac/mul//name//precipitation/def/'
    ),
    (
        'niger/chirps-precip-jas',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/(0)/(16)/RANGE/Y/(11)/(17)/RANGE/T/(Jul-Sep)/seasonalAverage/c%3A/3//units//months/def/%3Ac/mul//name//precipitation/def/'
    ),
    (
        'niger/chirps-precip-jjaso',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/(0)/(16)/RANGE/Y/(11)/(17)/RANGE/T/(Jun-Oct)/seasonalAverage/c%3A/5//units//months/def/%3Ac/mul/T//pointwidth/0/def/pop//name//precipitation/def/'
    ),
    (
        'niger/enacts-spi-jas',
        'http://iridl.ldeo.columbia.edu/home/.aaron/.Niger/.ENACTS/.seasonal/.rainfall/.CHIRPS/.SPI-3-month/.spi/T/(Jul-Sep)/VALUES/Y/first/17/RANGE/',
    ),
    (
        'niger/enacts-spi-jj',
        'http://iridl.ldeo.columbia.edu/home/.aaron/.Niger/.ENACTS/.seasonal/.rainfall/.CHIRPS/.SPI-2-month/.spi/T/(Jun-Jul)/VALUES/T//pointwidth/0/def/1.5/shiftGRID/Y/first/17/RANGE/',
    ),
    (
        'niger/chirp-spi-jj',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRP/.v1p0/.dekad/.prcp/X/0/15.975/RANGE/Y/11.025/17/RANGE/monthlyAverage/3./mul/2/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/2/gammaprobs/2/gammastandardize/T//pointwidth/2/def//defaultvalue/%7Blast%7Ddef/-0.5/shiftGRID/T/first/pointwidth/1/sub/add/last/RANGE//long_name/(Standardized%20Precipitation%20Index)/def/DATA/-3/3/RANGE/T/(Jun-Jul)/VALUES/T//pointwidth/0/def/1.5/shiftGRID/',
    ),
    (
        'niger/chirp-spi-jas',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRP/.v1p0/.dekad/.prcp/X/0/15.975/RANGE/Y/11.025/17/RANGE/monthlyAverage/3.0/mul/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7Ddef/(Jul-Sep)/VALUES//long_name/(Standardized%20Precipitation%20Index)/def/DATA/-3/3/RANGE/',
    ),
    (
        'niger/chirp-rainfall-sep',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRP/.v1p0/.dekad/.prcp/X/0/15.975/RANGE/Y/11.025/17/RANGE/monthlyAverage/3.0/mul/T/(Sep)/VALUES/T/-1/shiftGRID/',
    ),
    (
        'niger/chirps-dryspell',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p05/.prcp/X/(0)/(16)/RANGE/Y/(11)/(17)/RANGE/T/(1%20Jun)/61/1/(lt)/0.9/seasonalLLS/T//pointwidth/0/def/(months%20since%201960-01-01)/streamgridunitconvert/T/1.5/shiftGRID/'
    ),
    (
        'niger/chirps-onset',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p05/.prcp/X/(0)/(16)/RANGE/Y/(11)/(17)/RANGE/T/({start})/({end})/RANGE/T/(1%20May)/122/0.1/3/20/1/15/30/onsetDate/T/sub/T//pointwidth/0/def/(months%20since%201960-01-01)/streamgridunitconvert/T/3.5/shiftGRID/',
        (
            {'start': 1991, 'end': 2000},
            {'start': 2001, 'end': 2010},
            {'start': 2011, 'end': 2020},
            {'start': 2021, 'end': 2022},
        )
    ),
    (
        'niger/pnep-jas-v3',
        'https://iridl.ldeo.columbia.edu/home/.aaron/.Niger/.Forecasts/.NextGen/.PRCPPRCP_CCAFCST_JAS_v3/.NextGen/.FbF/.pne/S/(1%20Jan)/(1%20Feb)/(1%20Mar)/(1%20Apr)/(1%20May)/(1%20Jun)/VALUES/'
    ),
    (
        'niger/pnep-jj',
        'http://iridl.ldeo.columbia.edu/home/.remic/.Niger/.Forecasts/.NextGen/.PRCPPRCP_CCAFCST_JJ/.NextGen/.FbF/.pne/S/(1%20Jan)/(1%20Feb)/(1%20Mar)/(1%20Apr)/(1%20May)/(1%20Jun)/VALUES/'
    ),
    (
        'niger/pnep-jjaso',
        'http://iridl.ldeo.columbia.edu/home/.aaron/.Niger/.Forecasts/.NextGen/.PRCPPRCP_CCAFCST_JJASO/.NextGen/.FbF/.pne/S/(1%20Jan)/(1%20Feb)/(1%20Mar)/(1%20Apr)/(1%20May)/(1%20Jun)/VALUES/'
    ),
    (
        'niger/poe-onset',
        'http://iridl.ldeo.columbia.edu/home/.remic/.Niger/.Forecasts/.NextGen/.PRCPonset_CCAFCST_{part}/.NextGen/.FbF/.pne/100/exch/sub//name//poe/def//long_name/(Probability%2520of%2520exceedance)/def/S/(1%20Jan)/(1%20Feb)/(1%20Mar)/(1%20Apr)/(1%20May)/(1%20Jun)/VALUES/',
        (
            {'part': 'N'},
            {'part': 'S'},
        ),
    ),
    (
        'niger/poe-dryspell',
        'http://iridl.ldeo.columbia.edu/home/.remic/.Niger/.Forecasts/.NextGen/.PRCPmaxDS_CCAFCST_JA/.NextGen/.FbF/.pne/100/exch/sub//name//poe/def//long_name/(Probability%20of%20exceedance)/def/S/(1%20Jan)/(1%20Feb)/(1%20Mar)/(1%20Apr)/(1%20May)/(1%20Jun)/VALUES/'
    ),
    (
        'niger/subseas-dry-tercile-sjul-l15',
        '{url}',
        (
            {'url': 'http://iridl.ldeo.columbia.edu/SOURCES/.IRI/.FD/.SubX/.SubX_Hindcast/.Subx_biweekly_Precipitation/.MME_v21_Precipitation_ELR/.prob/L/15/VALUE/L/removeGRID/S/(0000%203%20Jul%201999)/(0000%201%20Jul%202000)/(0000%207%20Jul%202001)/(0000%206%20Jul%202002)/(0000%205%20Jul%202003)/(0000%203%20Jul%202004)/(0000%202%20Jul%202005)/(0000%201%20Jul%202006)/(0000%207%20Jul%202007)/(0000%205%20Jul%202008)/(0000%204%20Jul%202009)/(0000%203%20Jul%202010)/(0000%202%20Jul%202011)/(0000%207%20Jul%202012)/(0000%206%20Jul%202013)/(0000%205%20Jul%202014)/(0000%204%20Jul%202015)/(0000%202%20Jul%202016)/VALUES/C/(Below_Normal)/VALUE/C/removeGRID/X/0/16/RANGEEDGES/Y/11/17/RANGEEDGES/S/(months%20since%201960-01-01)/streamgridunitconvert/S/toi4/use_as_grid/S/1.5/shiftGRID/S//T/renameGRID/'},
            {'url': 'http://iridl.ldeo.columbia.edu/SOURCES/.IRI/.FD/.SubX/.SubX_Tercile_Forecast/.Subx_biweekly_Precipitation/.MMEv21_Precipitation_ELR/.prob/L/15/VALUE/L/removeGRID/C/(Below_Normal)/VALUE/C/removeGRID/S/(Jul)/VALUES/S/(1%20Jul)/VALUES/X/0/16/RANGEEDGES/Y/11/17/RANGEEDGES/S/(months%20since%201960-01-01)/streamgridunitconvert/S/toi4/use_as_grid/S/1.5/shiftGRID/S//T/renameGRID/'}
        ),
    ),
    (
        "niger/obs-subseas-rainfall",
        "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.UNIFIED_PRCP/.GAUGE_BASED/.GLOBAL/.v1p0/.extREALTIME/.rain/T/14/runningAverage/c%3A/14.0//units//days/def/%3Ac/mul/T/(1200%2010%20Jul%201999%20-%201200%2024%20Jul%201999)/(1200%20%208%20Jul%202000%20-%201200%2022%20Jul%202000)/(1200%2014%20Jul%202001%20-%201200%2028%20Jul%202001)/(1200%2013%20Jul%202002%20-%201200%2027%20Jul%202002)/(1200%2012%20Jul%202003%20-%201200%2026%20Jul%202003)/(1200%2010%20Jul%202004%20-%201200%2024%20Jul%202004)/(1200%20%209%20Jul%202005%20-%201200%2023%20Jul%202005)/(1200%20%208%20Jul%202006%20-%201200%2022%20Jul%202006)/(1200%2014%20Jul%202007%20-%201200%2028%20Jul%202007)/(1200%2012%20Jul%202008%20-%201200%2026%20Jul%202008)/(1200%2011%20Jul%202009%20-%201200%2025%20Jul%202009)/(1200%2010%20Jul%202010%20-%201200%2024%20Jul%202010)/(1200%20%209%20Jul%202011%20-%201200%2023%20Jul%202011)/(1200%2014%20Jul%202012%20-%201200%2028%20Jul%202012)/(1200%2013%20Jul%202013%20-%201200%2027%20Jul%202013)/(1200%2012%20Jul%202014%20-%201200%2026%20Jul%202014)/(1200%2011%20Jul%202015%20-%201200%2025%20Jul%202015)/(1200%20%209%20Jul%202016%20-%201200%2023%20Jul%202016)/(1200%20%208%20Jul%202017%20-%201200%2022%20Jul%202017)/(1200%2014%20Jul%202018%20-%201200%2028%20Jul%202018)/(1200%2013%20Jul%202019%20-%201200%2027%20Jul%202019)/(1200%2011%20Jul%202020%20-%201200%2025%20Jul%202020)/(1200%2010%20Jul%202021%20-%201200%2024%20Jul%202021)/(1200%20%209%20Jul%202022%20-%201200%2023%20Jul%202022)/VALUES/T//pointwidth/0/def/pop/T/(months%20since%201960-01-01)/streamgridunitconvert/T/toi4/use_as_grid/T/1.5/shiftGRID/X/0/16/RANGEEDGES/Y/11/17/RANGEEDGES/"
    ),
    (
        'niger/enacts-mon-spi-jj',
        'http://iridl.ldeo.columbia.edu/home/.rijaf/.Niger/.ENACTS/.MON/.seasonal/.rainfall/.CHIRP/.SPI-2-month/.spi/T/%28Jun-Jul%29VALUES/T/1.5/shiftGRID/Y/first/17/RANGE/',
    ),
    (
        'niger/enacts-mon-spi-jas',
        'http://iridl.ldeo.columbia.edu/home/.rijaf/.Niger/.ENACTS/.MON/.seasonal/.rainfall/.CHIRP/.SPI-3-month/.spi/T/%28Jul-Sep%29VALUES/Y/first/17/RANGE/',
    ),
    (
        'niger/enacts-mon-rainfall-sep',
        'http://iridl.ldeo.columbia.edu/home/.rijaf/.Niger/.ENACTS/.MON/.monthly/.rainfall/.CHIRP/.rfe_merged/T/(Sep)/VALUES/Y/first/17/RANGE/T/-1/shiftGRID/',
    ),
    (
        "niger/wrsi-jas",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.EROS/.FEWS/.dekadal/.WAF/.Millet/.do/X/0.125/15.875/RANGE/Y/11.125/16.875/RANGE/T/(Jul-Sep)/seasonalAverage/T/(months%20since%201960-01-01)/streamgridunitconvert/",
    ),
    (
        'mali/chirps-precip-jun-jas',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/(-12)/(4)/RANGE/Y/(10)/(25)/RANGE/T/(Jun)/seasonalAverage/T//pointwidth/0/def/2/shiftGRID/c%3A/1//units//months/def/%3Ac/mul//name//precipitation/def/'
    ),
    (
        'mali/chirps-precip-jas',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/(-12)/(4)/RANGE/Y/(10)/(25)/RANGE/T/(Jul-Sep)/seasonalAverage/c%3A/3//units//months/def/%3Ac/mul//name//precipitation/def/'
    ),
    (
        'mali/chirps-precip-jjaso',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/(-12)/(4)/RANGE/Y/(10)/(25)/RANGE/T/(Jun-Oct)/seasonalAverage/c%3A/5//units//months/def/%3Ac/mul/T//pointwidth/0/def/pop//name//precipitation/def/'
    ),
    (
        'mali/chirp-spi-jj-jas',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRP/.v1p0/.dekad/.prcp/X/-12/4/RANGE/Y/10/25/RANGE/monthlyAverage/3./mul/2/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/2/gammaprobs/2/gammastandardize/T//pointwidth/2/def//defaultvalue/%7Blast%7Ddef/-0.5/shiftGRID/T/first/pointwidth/1/sub/add/last/RANGE//long_name/(Standardized%20Precipitation%20Index)/def/DATA/-3/3/RANGE/T/(Jun-Jul)/VALUES/T//pointwidth/0/def/1.5/shiftGRID/',
    ),
    (
        'mali/chirp-spi-jas',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRP/.v1p0/.dekad/.prcp/X/-12/4/RANGE/Y/10/25/RANGE/monthlyAverage/3.0/mul/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7Ddef/(Jul-Sep)/VALUES//long_name/(Standardized%20Precipitation%20Index)/def/DATA/-3/3/RANGE/',
    ),
    (
        'mali/chirp-rainfall-sep-jas',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRP/.v1p0/.dekad/.prcp/X/-12/4/RANGE/Y/10/25/RANGE/monthlyAverage/3.0/mul/T/(Sep)/VALUES/T/-1/shiftGRID/',
    ),
    (
        'mali/chirps-dryspell-jas',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p05/.prcp/X/(-12)/(4)/RANGE/Y/(10)/(25)/RANGE/T/(1%20Jun)/61/1/(lt)/0.9/seasonalLLS/T//pointwidth/0/def/(months%20since%201960-01-01)/streamgridunitconvert/T/1.5/shiftGRID/'
    ),
    (
        'rain-guatemala',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/-92.5/.1/-88/GRID/Y/13/.1/18/GRID/T/(Oct-Dec)/seasonalAverage//name//prcp_est/def/',
    ),
    (
        'pnep-guatemala',
        'http://iridl.ldeo.columbia.edu/home/.xchourio/.ACToday/.CPT/.NextGen/.Seasonal/.CHIRPS/.GTM-FbF/.NextGen/.FbF/.pne/S/%281%20Sep%29VALUES/P/grid://name//P/def//units//percent/def/5/5/95/:grid/replaceGRID/L/removeGRID/'
    ),
    (
        'guatemala/rain-may-jja',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/-92.5/.1/-88/GRID/Y/13/.1/18/GRID/T/(May)/seasonalAverage//name//prcp_est/def/T/2/shiftGRID/',
    ),
    (
        'guatemala/rain-jj-jja',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/-92.5/.1/-88/GRID/Y/13/.1/18/GRID/T/(Jun-Jul)/seasonalAverage//name//prcp_est/def/T/.5/shiftGRID/',
    ),
    (
        'guatemala/rain-ja-jja',
     	'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/-92.5/.1/-88/GRID/Y/13/.1/18/GRID/T/(Jul-Aug)/seasonalAverage//name//prcp_est/def/T/-.5/shiftGRID/',
    ),
    (
        'guatemala/rain-jja',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/-92.5/.1/-88/GRID/Y/13/.1/18/GRID/T/(Jun-Aug)/seasonalAverage//name//prcp_est/def/',
    ),
    (
        'guatemala/pnep-jja',
        'http://iridl.ldeo.columbia.edu/home/.xchourio/.ACToday/.CPT/.NextGen/.Seasonal/.CHIRPS/.GTM-FbF/.NextGen/.FbF/.pne/S/%281%20May%29VALUES/P/grid://name//P/def//units//percent/def/5/5/95/:grid/replaceGRID/L/removeGRID/'
    ),
    (
        # Static (non-updating) snapshot of
        # http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NESDIS/.STAR/.VHP/.16km/.VHI
        # which is doubly broken: the upstram URL has changed, plus they had a
        # crash and lost all their data. Estimated time for restoring the data is
        # "a long time."
        'guatemala/vhi-jja',
        'https://iridl.ldeo.columbia.edu/home/.aaron/.nesdis/.VHP_16km/.VH/.VHI/X/-92.5/-88/RANGE/Y/13/18/RANGE/T/(1%20Jan%201996)/last/RANGE/T/(Jun-Aug)/seasonalAverage/',
    ),
   (
        'guatemala/rain-oct-aso',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/-92.5/.1/-88/GRID/Y/13/.1/18/GRID/T/(Oct)/seasonalAverage//name//prcp_est/def/T/-1/shiftGRID/',
    ),
    (
        'guatemala/rain-aso',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/-92.5/.1/-88/GRID/Y/13/.1/18/GRID/T/(Aug-Oct)/seasonalAverage//name//prcp_est/def/',
    ),
    (
        # Static (non-updating) snapshot of
        # http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NESDIS/.STAR/.VHP/.16km/.VHI
        # which is doubly broken: the upstram URL has changed, plus they had a
        # crash and lost all their data. Estimated time for restoring the data is
        # "a long time."
        'guatemala/vhi-aso',
        'https://iridl.ldeo.columbia.edu/home/.aaron/.nesdis/.VHP_16km/.VH/.VHI/X/-92.5/-88/RANGE/Y/13/18/RANGE/T/(1%20Jan%201996)/last/RANGE/T/(Aug-Oct)/seasonalAverage/',
    ),
   (
        'guatemala/rain-may',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/-92.5/.1/-88/GRID/Y/13/.1/18/GRID/T/(May)/seasonalAverage//name//prcp_est/def/',
    ),
    (
        # Static (non-updating) snapshot of
        # http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NESDIS/.STAR/.VHP/.16km/.VHI
        # which is doubly broken: the upstram URL has changed, plus they had a
        # crash and lost all their data. Estimated time for restoring the data is
        # "a long time."
        'guatemala/vhi-may',
        'https://iridl.ldeo.columbia.edu/home/.aaron/.nesdis/.VHP_16km/.VH/.VHI/X/-92.5/-88/RANGE/Y/13/18/RANGE/T/(1%20Jan%201996)/last/RANGE/T/(May)/seasonalAverage/',
    ),
   (
        'djibouti/rain-jas',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/41.625/43.375/RANGE/Y/10.875/12.875/RANGE/T/(Jul-Sep)/seasonalAverage/30/mul//units/(mm/month)/def/'
    ),
    (
        'djibouti/rain-mam',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/41.625/43.375/RANGE/Y/10.875/12.875/RANGE/T/(Mar-May)/seasonalAverage/30/mul//units/(mm/month)/def/'
    ),
    (
        'djibouti/rain-ond',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/41.625/43.375/RANGE/Y/10.875/12.875/RANGE/T/(Oct-Dec)/seasonalAverage/30/mul//units/(mm/month)/def/'
    ),
    (
        'djibouti/rain-jjas',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/41.625/43.375/RANGE/Y/10.875/12.875/RANGE/T/(Jun-Sep)/seasonalAverage/30/mul//units/(mm/month)/def/'
    ),
    (
        'djibouti/pnep-jas',
        'http://iridl.ldeo.columbia.edu/home/.aaron/.Djibouti/.PRCPPRCP_CCAFCST/.NextGen/.FbF/.pne/',
    ),
    (
        'djibouti/pnep-mam',
        'http://iridl.ldeo.columbia.edu/home/.aaron/.ICPAC/.Forecasts/.CPT/.Djibouti/.PRCP_MME/.NextGen/.FbF/.pne/',
    ),
    (
        "djibouti/ndvi-jas",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/41.625/.25/43.375/GRID/Y/10.875/.25/12.875/GRID/T/(Jul-Sep)/seasonalAverage/",
    ),
    (
        "djibouti/ndvi-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/41.625/.25/43.375/GRID/Y/10.875/.25/12.875/GRID/T/(Mar-May)/seasonalAverage/",
    ),
    (
        "djibouti/ndvi-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/41.625/.25/43.375/GRID/Y/10.875/.25/12.875/GRID/T/(Oct-Dec)/seasonalAverage/",
    ),
    (
        "djibouti/ndvi-jjas",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/41.625/.25/43.375/GRID/Y/10.875/.25/12.875/GRID/T/(Jun-Sep)/seasonalAverage/",
    ),
    (
        "djibouti/ndvi-viirs-jas",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/41.625/.25/43.375/GRID/Y/10.875/.25/12.875/GRID/T/(Jul-Sep)/seasonalAverage/",
    ),
    (
        "djibouti/ndvi-viirs-mam",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/41.625/.25/43.375/GRID/Y/10.875/.25/12.875/GRID/T/(Mar-May)/seasonalAverage/",
    ),
    (
        "djibouti/ndvi-viirs-ond",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/41.625/.25/43.375/GRID/Y/10.875/.25/12.875/GRID/T/(Oct-Dec)/seasonalAverage/",
    ),
    (
        "djibouti/ndvi-viirs-jjas",
        "https://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GSFC/.SED/.TISL/.LandSIPS/.VNP13/.C2/.v002/.NDVI/X/41.625/.25/43.375/GRID/Y/10.875/.25/12.875/GRID/T/(Jun-Sep)/seasonalAverage/",
    ),
    (
        "djibouti/spi-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/41.625/43.375/RANGE/Y/10.875/12.875/RANGE/monthlyAverage/T/(days%20since%201960-01-01)/streamgridunitconvert/T/differential_mul/T/(months%20since%201960-01-01)/streamgridunitconvert/a%3A/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7D/def/-1./shiftGRID/T/(Mar-May%201991)/last/12/RANGESTEP//long_name/(Standardized%20Precipitation%20Index)/def/%3Aa%3A/T/(Mar-May)/seasonalAverage/0./flaggt/%5BT%5Daverage/1./3./div/flaggt/1./masklt/%3Aa/mul/DATA/-3/3/RANGE//name//spi/def/"
    ),
    (
        "djibouti/spi-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/41.625/43.375/RANGE/Y/10.875/12.875/RANGE/monthlyAverage/T/(days%20since%201960-01-01)/streamgridunitconvert/T/differential_mul/T/(months%20since%201960-01-01)/streamgridunitconvert/a%3A/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7D/def/-1./shiftGRID/T/(Oct-Dec%201991)/last/12/RANGESTEP//long_name/(Standardized%20Precipitation%20Index)/def/%3Aa%3A/T/(Oct-Dec)/seasonalAverage/0./flaggt/%5BT%5Daverage/1./3./div/flaggt/1./masklt/%3Aa/mul/DATA/-3/3/RANGE//name//spi/def/"
    ),
    (
        "djibouti/spi-jjas",
        "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/41.625/43.375/RANGE/Y/10.875/12.875/RANGE/monthlyAverage/T/(days%20since%201960-01-01)/streamgridunitconvert/T/differential_mul/T/(months%20since%201960-01-01)/streamgridunitconvert/a%3A/4/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/4/gammaprobs/4/gammastandardize/T//pointwidth/4/def//defaultvalue/%7Blast%7D/def/-1.5/shiftGRID//long_name/(Standardized%20Precipitation%20Index)/def/%3Aa%3A/T/4/runningAverage/T/12/splitstreamgrid/0./flaggt/%5BT2%5Daverage/1./3./div/flaggt/1./masklt/%5BT%5D/REORDER/CopyStream/%3Aa/mul/DATA/-3/3/RANGE//name//spi/def/T/(Jun-Sep%201981)/last/12/RANGESTEP/",
    ),
    (
        "djibouti/spi-jas",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/41.625/43.375/RANGE/Y/10.875/12.875/RANGE/monthlyAverage/T/(days%20since%201960-01-01)/streamgridunitconvert/T/differential_mul/T/(months%20since%201960-01-01)/streamgridunitconvert/a%3A/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7D/def/-1./shiftGRID/T/(Jul-Sep%201991)/last/12/RANGESTEP//long_name/(Standardized%20Precipitation%20Index)/def/%3Aa%3A/T/(Jul-Sep)/seasonalAverage/0./flaggt/%5BT%5Daverage/1./3./div/flaggt/1./masklt/%3Aa/mul/DATA/-3/3/RANGE//name//spi/def/"
    ),
    (
        "djibouti/wrsi-ond",
        "https://iridl.ldeo.columbia.edu/SOURCES/.USGS/.EROS/.FEWS/.dekadal/.EAF/.short_rains_rangelands/.do/X/41.625/43.375/RANGE/Y/10.875/12.875/RANGE/T/(Oct-Dec)/seasonalAverage/",
    ),
    (
        "djibouti/wrsi-jjas",
        "https://iridl.ldeo.columbia.edu/SOURCES/.USGS/.EROS/.FEWS/.dekadal/.EAF/.long_rains_rangelands/.do/X/41.625/43.375/RANGE/Y/10.875/12.875/RANGE/T/(Jun-Sep)/seasonalAverage/",
    ),
    (
        "lesotho/pnep-djf",
        "http://iridl.ldeo.columbia.edu/home/.remic/.Lesotho/.Forecasts/.NextGen/.DJF_PRCPPRCP_CCAFCST/.NextGen/.FbF/.pne/",
    ),
    (
        "lesotho/enacts-precip-djf",
        "https://iridl.ldeo.columbia.edu/home/.aaron/.dle_lms/.Lesotho/.ENACTS/.ALL/.monthly/.rainfall/.rfe/T/(Dec-Feb)/seasonalAverage/3/mul/",
    ),
    (
        "lesotho/chirps-djf",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/26.925/29.5875/RANGE/Y/-30.7875/-28.425/RANGE/T/(Dec-Feb)/seasonalAverage/c%3A/3//units//months/def/%3Ac/mul//name//precipitation/def/",
    ),
    (
        "lesotho/ndvi-djf",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.SAF/.NDVI/X/26.925/.0375/29.5875/GRID/Y/-30.7875/.0375/-28.425/GRID/T/(Dec-Feb)/seasonalAverage/",
    ),
    (
        "lesotho/wrsi-djf",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.EROS/.FEWS/.dekadal/.SAF/.Maize/.do/X/26.925/29.5875/RANGE/Y/-30.7875/-28.425/RANGE/T/(Dec-Feb)/seasonalAverage/T/(months%20since%201960-01-01)/streamgridunitconvert/",
    ),
    (
        "lesotho/pnep-ond",
        "http://iridl.ldeo.columbia.edu/home/.remic/.Lesotho/.Forecasts/.NextGen/.OND_PRCPPRCP_CCAFCST/.NextGen/.FbF/.pne/",
    ),
    (
        "lesotho/pnep-ond-training",
        "http://iridl.ldeo.columbia.edu/0/home/.aaron/.dle_lms/.Lesotho/.Forecasts-training/.NextGen/.OND_PRCPPRCP_CCAFCST/.NextGen/.FbF/.pne/",
    ),
    (
        "lesotho/enacts-precip-ond",
        "https://iridl.ldeo.columbia.edu/home/.aaron/.dle_lms/.Lesotho/.ENACTS/.ALL/.monthly/.rainfall/.rfe/T/(Oct-Dec)/seasonalAverage/3/mul/",
    ),
    (
        "lesotho/chirps-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/26.925/29.5875/RANGE/Y/-30.7875/-28.425/RANGE/T/(Oct-Dec)/seasonalAverage/c%3A/3//units//months/def/%3Ac/mul//name//precipitation/def/",
    ),
    (
        "lesotho/ndvi-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.SAF/.NDVI/X/26.925/.0375/29.5875/GRID/Y/-30.7875/.0375/-28.425/GRID/T/(Oct-Dec)/seasonalAverage/",
    ),
    (
        "lesotho/wrsi-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.EROS/.FEWS/.dekadal/.SAF/.Maize/.do/X/26.925/29.5875/RANGE/Y/-30.7875/-28.425/RANGE/T/(Oct-Dec)/seasonalAverage/T/(months%20since%201960-01-01)/streamgridunitconvert/",
    ),
]


if __name__ == '__main__':
    os.umask(0o002)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cookiefile', type=os.path.expanduser)
    parser.add_argument(
        '--datadir',
        default='/data/aaron/fbf-candidate',
        type=os.path.expanduser,
    )
    parser.add_argument('datasets', nargs='*')
    opts = parser.parse_args()

    selected_url_datasets = opts.datasets or [ds[0] for ds in url_datasets]

    for dataset in url_datasets:
        name = dataset[0]
        pattern = dataset[1]
        if len(dataset) == 3:
            slices = dataset[2]
        else:
            slices = ({},)

        if name not in selected_url_datasets:
            continue
        print(name)
        for i, args in enumerate(slices):
            ncfilepath = f'{opts.datadir}/{name}-{i}.nc'
            leafdir = os.path.dirname(ncfilepath)

            if not os.path.exists(leafdir):
                os.makedirs(leafdir)

            if opts.cookiefile is None:
                cookieopt = ""
            else:
                cookieopt = f"-b {opts.cookiefile}"

            url = pattern.format(**args)
            os.system(f"curl {cookieopt} -o {ncfilepath} '{url}data.nc'")
            assert os.path.exists(ncfilepath)
        zarrpath = "%s/%s.zarr" % (opts.datadir, name)
        print("Converting to zarr")
        ds = pingrid.open_mfdataset([f'{opts.datadir}/{name}-{i}.nc' for i in range(len(slices))])
        # TODO do this in Ingrid
        if 'Y' in ds and ds['Y'][0] > ds['Y'][1]:
            ds = ds.reindex(Y=ds['Y'][::-1])
        if 'P' in ds:
            ds = ds.chunk({'P': 1})
        if os.path.exists(zarrpath):
            shutil.rmtree(zarrpath)
        ds.to_zarr(zarrpath)
