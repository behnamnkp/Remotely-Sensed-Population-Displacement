from osgeo import gdal, osr, gdal_array
from osgeo import *
import xarray as xr
import numpy as np
import os
import sys
# source https://www.linkedin.com/pulse/convert-netcdf4-file-geotiff-using-python-chonghua-yin/
gdal.GetDriverByName

def GetnetCDFInfobyName(in_filename, var_name):
    src_ds = gdal.Open(in_filename)
    if src_ds is None:
        print("Opening failed")
        sys.exit()

    if src_ds.GetSubDatasets() > 1:
        subdataset = 'NETCDF:"' + in_filename + '":' + var_name
        src_ds_sd = gdal.Dataset(subdataset)
        NDV = src_ds_sd.GetRasterBand(1).GetNoDataValue()
        xsize = src_ds_sd.RasterXSize
        ysize = src_ds_sd.RasterYSize
        GeoT = src_ds_sd.GetGeoTransform()
        Projection = osr.SpatialReference()
        Projection.ImportFromWkt(src_ds_sd.GetProjectionRef())
        src_ds_sd = None
        src_ds = None
        xr_resemble = xr.open_dataset(in_filename)
        data = xr_resemble[var_name]
        data = np.ma.masked_array(data, mask=data==NDV, fill_value=NDV)

        return NDV, xsize, ysize, GeoT, Projection, data

def create_geotiff(suffix, Array, NDV, xsize, ysize, GeoT, Projection):
    DataType = gdal_array.NumericTypeCodeToGDALTypeCode(Array.dtype)
    if type(DataType)!= np.int:
        if DataType.startswith('gdal.GDT_')==False:
            DataType=eval('gdal.GDT_' + DataType)

    NewFileName = suffix + '.tif'
    zsize = Array.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    Array[np.isnan(Array)] = NDV
    Dataset = driver.Create(NewFileName, xsize, ysize, zsize, DataType)
    Dataset.SetGeoTransform(GeoT)
    Dataset.SetProjection(Projection.ExportToWkt())
    for i in range(0, zsize):
        Dataset.GetRasterBand(i+1).WriteArray(Array[i])
        Dataset.GetRasterBand(i+1).SetNoDataValue(NDV)

    Dataset.FlushCache()
    return NewFileName

if __name__ == "__main__":
    infile = r'C:\Users\bnikparv\Downloads\test.tar\JRR-CloudDCOMP_v2r3_npp_s202101161800125_e202101161801366_c202101161842360.nc'
    var_name = 'CloudMicroVisOD'
    NDV, xsize, ysize, GeoT, Projection, data = GetnetCDFInfobyName(infile, var_name)
    outfile = create_geotiff(var_name, data, NDV, xsize, ysize, GeoT, Projection)


