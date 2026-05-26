"""
Examples to read and show the data
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from netCDF4 import Dataset
import cmocean
import warnings
import seaborn as sns

sns.set_style("ticks")

ssc_colors = ['#1c3e72', '#6b8fbe', '#ffffff', '#c89643', '#8f5c05' ]
cmap_spm = mcolors.LinearSegmentedColormap.from_list('water_sediment', ssc_colors)
VAR_CONFIG = {
    'chlorophyll': {
        'cmap': 'viridis',
        'vmin': 0.1, 'vmax': 100,
        'norm': 'log',             
        'label': 'Chl-a ($\mu$g L$^{-1}$)'
    },
    'spm': {
        'cmap': cmap_spm,
        'vmin': 0.1, 'vmax': 100,
        'norm': 'log',
        'label': 'SPM (mg L$^{-1}$)'
    },
    'Zsd': {
        'cmap': 'gist_earth_r',
        'vmin': 0.1, 'vmax': 10.0,
        'norm': 'log',
        'label': 'Zsd (m)'
    }
}
LAKES_BBOX = {
    'Selin Co':   [88.40, 89.40, 31.30, 32.2],
    'Bosten':     [86.60, 87.40, 41.80, 42.15],
    'Fuxian':     [102.75, 103.00, 24.30, 24.70],
    'Taihu':      [119.80, 120.70, 30.90, 31.60],
    'Chagan':     [124.10, 124.45, 45.10, 45.40]
}

def plot_lakes_monthly_matrix(variable, base_dir, out_dir, year=2024):
    """
    Plot the data
    """
    os.makedirs(out_dir, exist_ok=True)
    months = [5, 6, 7, 8, 9, 10]
    lake_names = list(LAKES_BBOX.keys())

    cfg = VAR_CONFIG[variable]
    
    fig = plt.figure(figsize=(14, 16), dpi=330)
    gs = GridSpec(len(months) + 1, len(lake_names), height_ratios=[1, 1, 1, 1, 1, 1, 0.15], wspace=0.05, hspace=0.05)
    
    im_for_cbar = None
    
    for row, month in enumerate(months):
        month_str = f"{month:02d}"
        nc_file = os.path.join(base_dir, f"CLWOP_VIIRS_{year}M{month_str}_monthly_v2.nc")
        if not os.path.exists(nc_file):
            print(f"Warning: Missing data for {year}-{month_str}")
            continue
            
        with Dataset(nc_file, 'r') as nc:
            lons = nc['lon'][:]
            lats = nc['lat'][:]
            data = nc[variable][:]
            data = np.where(data == -32767.0, np.nan, data)
            data[data<0.01] = np.nan
            for col, lake in enumerate(lake_names):
                ax = fig.add_subplot(gs[row, col])
                lon_min, lon_max, lat_min, lat_max = LAKES_BBOX[lake]
                lon_mask = (lons >= lon_min) & (lons <= lon_max)
                lat_mask = (lats >= lat_min) & (lats <= lat_max)
                subset_lon = lons[lon_mask]
                subset_lat = lats[lat_mask]
                subset_data = data[lat_mask, :][:, lon_mask]
                if cfg.get('norm') == 'log':
                    subset_data = np.where(subset_data <= 0, np.nan, subset_data)
                    norm = mcolors.LogNorm(vmin=cfg['vmin'], vmax=cfg['vmax'])
                    im = ax.pcolormesh(
                        subset_lon, subset_lat, subset_data, 
                        cmap=cfg['cmap'], norm=norm, shading='auto'
                    )
                else:
                    im = ax.pcolormesh(
                        subset_lon, subset_lat, subset_data, 
                        cmap=cfg['cmap'], vmin=cfg['vmin'], vmax=cfg['vmax'], 
                        shading='auto'
                    )
                if im_for_cbar is None:
                    im_for_cbar = im

              
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor('gray')
                
                if row == 0:
                    ax.set_title(lake, fontsize=18, pad=15)
                    
                if col == 0:
                    month_names = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
                    ax.set_ylabel(f"{month_names[row]}", fontsize=18, labelpad=15)

    if im_for_cbar is not None:
        cbar_ax = fig.add_subplot(gs[-1, :])
        box = cbar_ax.get_position()
        box.x0 = box.x0 + 0.2
        box.x1 = box.x1 - 0.2
        cbar_ax.set_position(box)
        cbar = fig.colorbar(im_for_cbar, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(cfg['label'], fontsize=16, labelpad=10)
        cbar.ax.tick_params(labelsize=14,)
    out_file = os.path.join(out_dir, f"Fig_Lakes_Matrix_2024_{variable}.png")
    plt.savefig(out_file, dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    netcdf_dir = r"E:\Satellitedata\CLWOP"
    output_dir = r"figures/"
    plot_lakes_monthly_matrix('chlorophyll', netcdf_dir, output_dir)
    plot_lakes_monthly_matrix('spm', netcdf_dir, output_dir)
    plot_lakes_monthly_matrix('Zsd', netcdf_dir, output_dir)
