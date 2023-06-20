from osgeo import gdal, ogr, osr
from scipy import ndimage
import geopandas as gp
import json

def resample_raster(inpute_path, output_path, layer, dx, dy, method='Majority'):

    # Open the input raster file
    input_ds = gdal.Open(inpute_path + layer)

    # Define the output raster properties
    output_size_x = dx  # Desired output width in pixels
    output_size_y = dy  # Desired output height in pixels

    if method == 'Majority':

        # Read the input raster as a NumPy array
        input_array = input_ds.ReadAsArray()

        # Calculate the new dimensions based on the desired pixel sizes
        new_width = int(input_ds.RasterXSize * input_ds.GetGeoTransform()[1] / output_size_x)
        new_height = int(input_ds.RasterYSize * abs(input_ds.GetGeoTransform()[5]) / output_size_y)

        # Resample the input array using the majority value
        resampled_array = ndimage.zoom(
            input_array, (new_height / input_array.shape[0], new_width / input_array.shape[1]), order=0)

        driver = gdal.GetDriverByName('GTiff')
        output_ds = driver.Create(output_path + 'rsm' + layer, new_width, new_height, 1, gdal.GDT_Float32)

        # Calculate the new geotransform for the output raster
        new_geotransform = list(input_ds.GetGeoTransform())
        new_geotransform[1] = input_ds.GetGeoTransform()[1] * input_ds.RasterXSize / new_width
        new_geotransform[5] = input_ds.GetGeoTransform()[5] * input_ds.RasterYSize / new_height

        # Set the geotransform and projection for the output raster
        output_ds.SetGeoTransform(new_geotransform)
        output_ds.SetProjection(input_ds.GetProjection())

        # Write the resampled array to the output raster
        output_band = output_ds.GetRasterBand(1)
        output_band.WriteArray(resampled_array)

        # Clean up and close the datasets
        output_band = None
        output_ds = None

    elif method=='GRIORA_Bilinear':

        # Resample the raster using gdal.Warp
        gdal.Warp(output_path + 'rsm' + layer, input_ds, xRes=output_size_x, yRes=output_size_y,
                  resampleAlg=gdal.GRIORA_Bilinear)

    elif method=='GRIORA_NearestNeighbour':

        # Resample the raster using gdal.Warp
        gdal.Warp(output_path + 'rsm' + layer, input_ds, xRes=output_size_x, yRes=output_size_y,
                  resampleAlg=gdal.GRIORA_NearestNeighbour)

    elif method=='GRIORA_Cubic':

        # Resample the raster using gdal.Warp
        gdal.Warp(output_path + 'rsm' + layer, input_ds, xRes=output_size_x, yRes=output_size_y,
                  resampleAlg=gdal.GRIORA_Cubic)

    elif method=='GRIORA_CubicSpline':

        # Resample the raster using gdal.Warp
        gdal.Warp(output_path + 'rsm' + layer, input_ds, xRes=output_size_x, yRes=output_size_y,
                  resampleAlg=gdal.GRIORA_CubicSpline)

    elif method=='GRIORA_Lanczos':

        # Resample the raster using gdal.Warp
        gdal.Warp(output_path + 'rsm' + layer, input_ds, xRes=output_size_x, yRes=output_size_y,
                  resampleAlg=gdal.GRIORA_Lanczos)

    elif method=='GRIORA_Average':

        # Resample the raster using gdal.Warp
        gdal.Warp(output_path + 'rsm' + layer, input_ds, xRes=output_size_x, yRes=output_size_y,
                  resampleAlg=gdal.GRIORA_Average)

    else:
        print(method + ' is not valid')

    # Close the raster dataset
    input_ds = None

def vectorize_raster(input_path, output_path, layer):
    """
    This function vectorizes a raster layer into areal units with the same size as in the raster layer.
    :return:
    """

    # Open the raster file
    raster_ds = gdal.Open(input_path + layer)

    # Retrieve raster properties
    width = raster_ds.RasterXSize
    height = raster_ds.RasterYSize
    pixel_size_x = raster_ds.GetGeoTransform()[1]
    pixel_size_y = raster_ds.GetGeoTransform()[5]
    projection = raster_ds.GetProjection()

    # Set up the output vector file
    driver = ogr.GetDriverByName('ESRI Shapefile')
    vector_ds = driver.CreateDataSource(output_path + layer.split('.')[0] + '.shp')
    layer_name = layer.split('.')[0]
    layer = vector_ds.CreateLayer(layer_name, geom_type=ogr.wkbPolygon, srs=osr.SpatialReference(projection))

    # Set the spatial reference of the vector layer
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(projection)

    # Create the field for the geometry
    id = ogr.FieldDefn('ID', ogr.OFTInteger)
    layer.CreateField(id)
    val = ogr.FieldDefn('Value', ogr.OFTInteger)
    layer.CreateField(val)
    area = ogr.FieldDefn('Area', ogr.OFTInteger)
    layer.CreateField(area)

    # Define the extent of the vector grid
    x_min = raster_ds.GetGeoTransform()[0]
    y_max = raster_ds.GetGeoTransform()[3]

    # Create the polygons
    for row in range(height):
        for col in range(width):
            x = x_min + col * pixel_size_x
            y = y_max + row * pixel_size_y

            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(x, y)
            ring.AddPoint(x + pixel_size_x, y)
            ring.AddPoint(x + pixel_size_x, y + pixel_size_y)
            ring.AddPoint(x, y + pixel_size_y)
            ring.AddPoint(x, y)

            polygon = ogr.Geometry(ogr.wkbPolygon)
            polygon.AddGeometry(ring)

            # Set the spatial reference for the geometry
            polygon.AssignSpatialReference(spatial_ref)

            # Create a feature and add the polygon geometry to it
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetGeometry(polygon)

            # Set the field value
            feature.SetField('ID', row * width + col)

            # Retrieve the raster value for the current pixel
            band = raster_ds.GetRasterBand(1)
            value = band.ReadAsArray(col, row, 1, 1)[0][0]

            # Set the raster value as an attribute
            feature.SetField('Value', int(value))

            # Calculate and set the pixel area as an attribute
            pixel_area = abs(pixel_size_x) * abs(pixel_size_y)
            feature.SetField('Area', pixel_area)

            # Add the feature to the layer
            layer.CreateFeature(feature)

    # Clean up and close the files
    feature = None
    vector_ds = None
    raster_ds = None
