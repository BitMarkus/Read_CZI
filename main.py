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
# Larger earlier images: x=16, y=21
# Smaller later images: x=15, y=20 
NUM_TILES_X = 15    # <- CHANGE HERE!!!
NUM_TILES_Y = 20    # <- CHANGE HERE!!!

# IMPORT SETTINGS #
# Scale factor when mosaic image is imported
# Scale factor for slices = original size
IMPORT_SCALE_SLICE = 1 
# Scale factor for the full overview image = 1/4 size
IMPORT_SCALE_FULL = 0.25 
# Image channel to import
# Currently only for a mosaic image with one channel
IMPORT_CHANNEL = 0

# EXPORT SETTINGS #
# Percentile settings
PERC_MIN = 0.5 # 0 for no change
PERC_MAX = 99.5 # 100 for no change
# Size of the image slices in pixel
SLICE_SIZE_X = 850
SLICE_SIZE_Y = 850
# Size of the final images for training (.png, grayscale, 8-bit)
RESIZE_X = 512
RESIZE_Y = 512

# PATHS AND SAVE #
# Path to mosaic image file
CZI_PATH = "czi/"
CZI_EXTENSION = ".czi"
# Path to exported images
EXPORT_PATH = "export/"
EXPORT_1X_PATH = "1x/"
EXPORT_4X_PATH = "4x/"
EXPORT_FULL_PATH = "full/"
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
        print(f"Folder {EXPORT_PATH + file_name + '/'} for exported images created.")

    #####################
    # EXPORT FULL IMAGE #
    #####################

    # Create an overview image in 25% size of the original
    print(f"Export of full image {file_name} at {IMPORT_SCALE_FULL*100}% size is starting. Please wait...")
    fcn.export_full_img(img_data, 
                        file_name, 
                        IMPORT_CHANNEL, 
                        IMPORT_SCALE_FULL, 
                        PERC_MIN, 
                        PERC_MAX, 
                        EXPORT_PATH, 
                        EXPORT_FULL_PATH, 
                        SAVE_IMG)             
    # Message after export is finished
    print(f"Export of full image finished.", end="")
    if(SAVE_IMG):        
        print(f" PNG image was saved to {EXPORT_PATH + file_name + '/' + EXPORT_FULL_PATH}.")
    else:
        print(" No image was saved!")
        
    ####################
    # EXPORT 1x SLICES #
    ####################
  
    print(f"Export of 1x slices for image {file_name} is starting. Please wait...")  
    fcn.export_sliced_img('1x',
                          img_data, 
                          file_name, 
                          IMPORT_CHANNEL,                     
                          SLICE_SIZE_X,
                          SLICE_SIZE_Y,
                          NUM_TILES_X,
                          NUM_TILES_Y,
                          RESIZE_X,
                          RESIZE_Y, 
                          IMPORT_SCALE_SLICE, 
                          PERC_MIN, 
                          PERC_MAX, 
                          EXPORT_PATH, 
                          EXPORT_1X_PATH, 
                          EXPORT_4X_PATH, 
                          save_img=True)   
    # Message for 4x slices
    print(f"Export of 1x slices finished.", end="")
    if(SAVE_IMG):        
        print(f" PNG images were saved to {EXPORT_PATH + file_name + '/' + EXPORT_1X_PATH}.")
    else:
        print(" No images were saved!")   

    ####################
    # EXPORT 4x SLICES #
    ####################
  
    print(f"Export of 4x slices for image {file_name} is starting. Please wait...")  
    fcn.export_sliced_img('4x',
                          img_data, 
                          file_name, 
                          IMPORT_CHANNEL,                     
                          SLICE_SIZE_X,
                          SLICE_SIZE_Y,
                          NUM_TILES_X,
                          NUM_TILES_Y,
                          RESIZE_X,
                          RESIZE_Y, 
                          IMPORT_SCALE_SLICE, 
                          PERC_MIN, 
                          PERC_MAX, 
                          EXPORT_PATH,
                          EXPORT_1X_PATH, 
                          EXPORT_4X_PATH, 
                          save_img=True)   
    # Message for 4x slices
    print(f"Export of 4x slices finished.", end="")
    if(SAVE_IMG):        
        print(f" PNG images were saved to {EXPORT_PATH + file_name + '/' + EXPORT_4X_PATH}.")
    else:
        print(" No images were saved!")   
       
    # Free memory
    del img_data
      
    print(f">> PROCESSING  OF IMAGE {file_name} FINISHED!")
    