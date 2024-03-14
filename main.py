# https://github.com/AllenCellModeling/aicspylibczi

import aicspylibczi
import pathlib
# Import own modules
import fcn

############
# SETTINGS #
############

# MOSAIC IMAGE DATA #
# Number of tiles
NUM_TILES = {'x': 10, 'y': 13}      # <- CHANGE HERE!!!
# Image channel to import
# Current version can only handle mosaic images with exactly 3 channels
# Ch 0 = Lysosomes (Lysotracker green -> 488 nm) in red
# Ch 1 = Cell membrane (WGA -> 647 nm) in green
# Ch 2 = Brightfield (DIC) in blue
NUM_CHANNELS = 3                    # <- CHANGE HERE!!!
# Determines which sizes of images are exported
EXPORT_TYPE = {'full': True, '1x': True, '4x': True, '16x': True}      # <- CHANGE HERE!!!

# IMPORT SETTINGS #
# Scale factor when mosaic image is imported
# Scale factor for slices = original size
IMPORT_SCALE_SLICE = 1 
# Scale factor for the full overview image = 1/4 size
IMPORT_SCALE_FULL = 0.25 

# EXPORT SETTINGS #
# Percentile settings for each channel as a dict in a dict
# min: 0 for no change, max: 100 for no change
PERC_NORM = {0: {'min': 0.1, 'max': 99.9}, 1: {'min': 0.1, 'max': 99.9}, 2: {'min': 0.5, 'max': 99.5}} 
# Size of the different image slices in pixel
SLICE_SIZE = {'1x': {'x': 2048, 'y': 2048}, '4x': {'x': 1024, 'y': 1024}, '16x': {'x': 512, 'y': 512},}
# Size of the final images for training (.png, rgb, 8-bit)
SLICE_RESIZE = {'x': 512, 'y': 512}

# PATHS AND SAVE #
# Path to mosaic image file
CZI_PATH = "czi/"
CZI_EXTENSION = ".czi"
# Path to exported images
EXPORT_PATH = "export/"
EXPORT_TYPE_PATH = {'full': 'full/', '1x': '1x/', '4x': '4x/', '16x': '16x/'}

# Debug mode
# Bool if images are saved and folders are created
SAVE_IMG = True

###########################
# CHECK IMAGES TO PROCESS #
###########################

# Get a list with all czi files in czi/ folder
# without path and extension
czi_list = fcn.get_czi_file_list(CZI_PATH, CZI_EXTENSION)
# print(czi_list)
  
# Iterate over file list
for file_name in czi_list: 

    print(f"\n>> PROCESSING IMAGE {file_name}:")
    
    ########################
    # LOAD CZI MOSAIC DATA #
    ########################

    # Load mosaic file
    img_pth = CZI_PATH + file_name + CZI_EXTENSION
    img_data = aicspylibczi.CziFile(pathlib.Path(img_pth))
    # fcn.get_czi_info(img_data)
    print(f"Loading of mosaic .czi image {file_name} successful.")
    
    ######################
    # SETUP IMAGE FOLDER #
    ######################
    
    if(SAVE_IMG):      
        # Create directory for each czi image 
        fcn.create_export_folder(EXPORT_PATH, file_name)

    #####################
    # EXPORT FULL IMAGE #
    #####################

    # Create an overview image in 25% size of the original
    fcn.export_full_img(img_data, 
                        EXPORT_TYPE, 
                        file_name, 
                        NUM_CHANNELS, 
                        IMPORT_SCALE_FULL, 
                        PERC_NORM, 
                        EXPORT_PATH, 
                        EXPORT_TYPE_PATH, 
                        SAVE_IMG)             

    #################
    # EXPORT SLICES #
    #################
 
    fcn.slice_tiles(img_data,
                    EXPORT_TYPE, 
                    file_name, 
                    NUM_CHANNELS, 
                    NUM_TILES, 
                    SLICE_SIZE, 
                    PERC_NORM, 
                    SLICE_RESIZE, 
                    IMPORT_SCALE_SLICE, 
                    EXPORT_PATH, 
                    EXPORT_TYPE_PATH, 
                    SAVE_IMG)

    # Free memory
    del img_data
      
    print(f">> PROCESSING OF IMAGE {file_name} FINISHED!")
     
    