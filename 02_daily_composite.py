"""
Using SNAP and GPT tool to generate daily composite of products.
"""
import os
from glob import glob
from datetime import datetime
from mapping_processing import create_xml
from nc_read import *
from calendar import isleap
from mapping_processing import mosaic
# mosaic

for year in range(2012, 2026):
    days = 366 if isleap(year) else 365
    unmapped_dir = rf'F:\CLARA\{year}'
    out_dir = rf'F:\CLARA_Daily_SNAP\{year}'
    os.makedirs(out_dir, exist_ok=True)

    for d in range(days):
        doy = str(int(d+1)).zfill(3)
        dt = datetime.strptime(str(year) + str(doy), '%Y%j').date()
        fmt = '%Y%m%d'
        date = dt.strftime(fmt)
        mosaic_file = os.path.join(out_dir, f'V{date}_CLARA_mosaic.dim')
        if os.path.exists(mosaic_file):
            print(f'{mosaic_file} exists, skip.')
            continue
        zsd_files = glob(os.path.join(unmapped_dir, f'V{year}{doy}*_CLARA.nc'))
        if len(zsd_files) < 1:
            continue
        #
        south, north = 23.5, 49.5
        west, east = 76.7, 133.1
        xml_file = os.path.splitext(mosaic_file)[0] + '.xml'
        limit = [south, west, north, east]
        l2prod_list = ['chlorophyll_a', 'Zsd', 'spm']
        limit = [south, west, north, east]
        create_xml(xml_file, l2prod_list, limit, resolution=750)
        # 2. mosaic operation
        mosaic(' '.join(zsd_files), xml_file, mosaic_file)
        if os.path.exists(xml_file): os.remove(xml_file)
