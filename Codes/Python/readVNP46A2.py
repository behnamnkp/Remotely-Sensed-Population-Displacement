# Use this code to convert hdf to .geotif
# make sure you enter the directory for input and output
# you need to install gdal library if you do not have it
import gdal, os
# You need to use conda environment to run this

## Set current directory to the directory where you have all hdf files
os.chdir('C:/wgetdown/MosulVNP2/raw2012')#: e.g. 'G:/sigspatial/HDFSLK/'
# read all file names. Make sure you do not have other file types in the directory
rasterFiles = os.listdir(os.getcwd())

fileNumber = len(rasterFiles)
for i in range(0, fileNumber):
    #Get File Name Prefix
    rasterFilePre = rasterFiles[i][:-3]
    fileExtension = ".tif"

    ## Open HDF file
    hdflayer = gdal.Open(rasterFiles[i], gdal.GA_ReadOnly)

    # The hdf file has several layers. The raw night light data is in layer 4, but
    # keep in mind that you may need to create separate .tif files for other layers
    # Here are some example layers:
    # 0: VNP_Grid_DNB:DNB_BRDF-Corrected_NTL
    # 1: VNP_Grid_DNB: VNP_Grid_DNB:DNB_Lunar_Irradiance
    # 2: VNP_Grid_DNB:VNP_Grid_DNB:Gap_Filled_DNB_BRDF-Corrected_NTL
    # 3: VNP_Grid_DNB:VNP_Grid_DNB:Mandatory_Quality_Flag
    # 4: VNP_Grid_DNB:VNP_Grid_DNB:Latest_High_Quality_Retrieval
    # 5: VNP_Grid_DNB:VNP_Grid_DNB:Snow_Flag
    # 6: VNP_Grid_DNB:VNP_Grid_DNB:QF_Cloud_Mask
    #24 23
    subhdflayer = hdflayer.GetSubDatasets()[0][0]
    rlayer = gdal.Open(subhdflayer, gdal.GA_ReadOnly)

    #setting appropriate name for output file
    outputName = subhdflayer[92:]#92
    outputNameNoSpace = outputName.strip().replace(" ","_").replace("/","_").replace("-","_")
    outputNameFinal = outputNameNoSpace + rasterFilePre.replace(".","_") + fileExtension

    # your output folder directory
    outputFolder = "C:/wgetdown/MosulVNP2/extract2012/brdf/" #e.g. G:/sigspatial/HDFSLK2TIF/
    outputRaster = outputFolder + outputNameFinal

    #collect bounding box coordinates and setting the coordinate system
    WestBoundCoord = rlayer.GetMetadata_Dict()["WestBoundingCoord"]
    EastBoundCoord = rlayer.GetMetadata_Dict()["EastBoundingCoord"]
    NorthBoundCoord = rlayer.GetMetadata_Dict()["NorthBoundingCoord"]
    SouthBoundCoord = rlayer.GetMetadata_Dict()["SouthBoundingCoord"]

    EPSG = "-a_srs EPSG:4326" #WGS84

    translateOptionText = EPSG+" -a_ullr " + WestBoundCoord + " " + NorthBoundCoord + " " + EastBoundCoord + " " + SouthBoundCoord

    translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine(translateOptionText))
    gdal.Translate(outputRaster,rlayer, options=translateoptions)

    print(rasterFilePre)