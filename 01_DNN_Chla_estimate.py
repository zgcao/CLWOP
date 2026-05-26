"""
Using the DNN model to generate Chl-a via Cao et al. (2024) approach
"""
import os
import joblib
import numpy as np
from glob import glob
from ncwrite import nc_write
from netCDF4 import Dataset
from tensorflow.keras.models import load_model
from datetime import datetime
import time
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# use cpu rather than GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def imread_noaa_l2(ncfile):
    nc = Dataset(ncfile, 'r')
    dts = ['nLw_486', 'nLw_551', 'nLw_638_ag', 'nLw_671', 'nLw_745', 'nLw_862']
    for i, dt in enumerate(dts):
        nLw = nc[f'geophysical_data/{dt}']
        ed0 = getattr(nc['geophysical_data'].variables[dt], 'solar_irradiance')
        if i == 0:
            data = np.zeros((nLw.shape[0], nLw.shape[1], len(dts)))
        rrs = nLw / ed0
        data[:, :, i] = rrs
    # read lat and lon
    lat = nc['navigation_data/latitude']
    lon = nc['navigation_data/longitude']
    return data, lat, lon


def img_est_chl(data, model, x_scaler, y_scaler):
    """Takes any number of input bands (shaped [Height, Width]) and
    returns the products for that image, in the same shape."""
    expected_features = 6
    assert (data.shape[-1] == expected_features), (
        f'Got {data.shape[-1]} features; expected {expected_features} features for VIIRS sensor')
    im_shape = data.shape[:-1]
    im_data = data.reshape((-1, expected_features))
    x_test = x_scaler.transform(im_data)
    # x_test = np.nan_to_num(x_test)
    # y_hat = model.predict(x_test).reshape(-1, 1)
    # 加速
    y_hat = model(x_test, training=False)
    est = np.exp(y_scaler.inverse_transform(y_hat))
    chl = est.reshape(im_shape)
    chl[chl < 0.01] = np.nan
    chl[chl > 200] = np.nan
    return chl


if __name__ == '__main__':
    # loading models and scalers for new tf version
    x_scaler = joblib.load('benchmarking/viirs_rrs_x_scaler_v3.pkl')
    y_scaler = joblib.load('benchmarking/viirs_chl_scaler_v3.pkl')
    dnn_model = load_model('benchmarking/modelsviirsn_chla_model_dnn_v3.keras')
    for year in range(2012, 2026):
        input_dir = rf'F:\CLARA_Daily_SNAP\msl12\{year}'
        out_dir = rf'F:\CLARA_Daily_SNAP\chla\{year}'
        os.makedirs(out_dir, exist_ok=True)
        nc_files = glob(input_dir + os.path.sep + 'V*_NPP_SCINSW_L2.nc')
        for i, nc_file in enumerate(nc_files):
            # doy: NorthernEast: 03, >07:TP and XJ; other: IMXL and EPL
            base_file = os.path.basename(nc_file)
            doy = int(base_file[5:8])
            hour = int(base_file[8:10])
            out_file = os.path.join(out_dir, base_file.replace('.nc', '_Chla.nc'))
            if os.path.exists(out_file):
                print(out_file + ' existing. skip...')
                continue
            data, lat, lon = imread_noaa_l2(nc_file)
            #
            time_s = time.process_time()
            chl_dnn = img_est_chl(data, dnn_model, x_scaler, y_scaler)
            time_e = time.process_time()
            print(time_e-time_s)
            nc_write(out_file, 'chla_dnn', np.rot90(chl_dnn, 2), new=True)
            nc_write(out_file, 'latitude', np.rot90(lat, 2), new=False)
            nc_write(out_file, 'longitude', np.rot90(lon, 2), new=False)
            #
            print('> {}/{}: {} has been written at {}'.format(i+1, len(nc_files), out_file, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
