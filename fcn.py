# https://forum.image.sc/t/reading-czi-file-in-python/39768/11
"""
Information CZI Dimension Characters:
- '0': 'Sample',  # e.g. RGBA
- 'X': 'Width',
- 'Y': 'Height',
- 'C': 'Channel',
- 'Z': 'Slice',  # depth
- 'T': 'Time',
- 'R': 'Rotation',
- 'S': 'Scene',  # contiguous regions of interest in a mosaic image
- 'I': 'Illumination',  # direction
- 'B': 'Block',  # acquisition
- 'M': 'Mosaic',  # index of tile for compositing a scene
- 'H': 'Phase',  # e.g. Airy detector fibers
- 'V': 'View',  # e.g. for SPIM
"""

import aicspylibczi
import numpy as np
from PIL import Image
import glob
import os

# Get image information
def get_czi_info(czi_data):
    # Image type
    print("Type:", type(czi_data))
    # Image dimensions, e.g.'STCZMYX'
    print("Dimensions:", czi_data.dims)
    # Image size, e.g. (1, 1, 336, 2208, 2752) (time?, channel, no tiles, height, width)
    print("Size:", czi_data.size)
    # Image shape, e.g. [{'X': (0, 2752), 'Y': (0, 2208), 'C': (0, 1), 'M': (0, 336), 'S': (0, 1)}]
    print("Shape:", czi_data.get_dims_shape())
    # If file is a mosaic image
    print("Is mosaic:", czi_data.is_mosaic(), "\n")

# Get numpy array information
def get_nparr_info(np_arr):
    # Data type
    print("Type:", type(np_arr))
    # Get number of dimensions
    print("Dimensions:", np_arr.ndim)
    # Get shape
    print("Shape:", np_arr.shape)
    # Get data type
    print("Data type:", np_arr.dtype)
    # Min and max values
    print("Min val:", np.min(np_arr))
    print("Max val:", np.max(np_arr))
    # Get item size
    print("Item size:", np_arr.itemsize)
    # Get total size
    print("Total size:", np_arr.nbytes)
    # Get amoumt of elements
    print("Elements:", np_arr.size, "\n")

# Reads a czi mosaic file
# Mosaic files ignore the S dimension and use an internal mIndex to reconstruct, 
# the scale factor allows one to generate a manageable image
# the C channel has been specified S & M are used internally for position so this is (T, Z, Y, X)
def read_czi_mosaic(czi_data, channel=0, scale=1):
    return czi_data.read_mosaic(C=channel, scale_factor=scale)

# Slice np image
def slice_np_img(np_arr, channel, start_x, start_y, size_x, size_y):
    slice_start = {'x': start_x, 'y': start_y}
    slice_end = {'x': slice_start['x'] + size_x, 'y': slice_start['y'] + size_y}
    return np_arr[channel, slice_start['y']:slice_end['y'], slice_start['x']:slice_end['x']]

# Normalize np image to 0-1
# Additionally percentile normalization is applied
def norm_by(x, min_, max_):
    norms = np.percentile(x, [min_, max_])
    img = np.clip((x - norms[0]) / (norms[1] - norms[0]), 0, 1)
    return img

# Convert normalized numpy array (0-1) in uint8 (0-255)
def convert_to_8bit(np_arr):
    np_arr *= 255
    return np_arr.astype(np.uint8)

# Convert numpy array to pillow image
# If the mode parameter is not specified, the mode is inferred 
# from the shape of the input array. If the input array is two-dimensional, 
# the mode is ‘L’ (grayscale), and if it is three-dimensional, the mode is ‘RGB’
def convert_np_to_pil(np_arr, mode="L"):
    # If the numpy array wasn't sliced, it has one more dimension (first, channel?)
    # which needs to be deleted in order for the numpy array to be converted to a pillow image
    if(np_arr.ndim == 3):
        np_arr = np_arr[0, :, :]
    return Image.fromarray(np_arr, mode=mode)

# Rezize numpy image
# Convert to Pillow type, resize and convert back to numpy array
def resize_np_img(np_arr, size_x, size_y):
    pil_img = convert_np_to_pil(np_arr)
    pil_resize = pil_img.resize((size_x, size_y))
    return np.array(pil_resize)

# Save image
def save_np_img(np_arr, export_path, img_name):
    pil_img = convert_np_to_pil(np_arr)
    pil_img.save(export_path + img_name)
    return True

# Processes the slices from the mosaic file
# Input must be a numpy array
# 1. Norms np Image to 0-1 and add percentile:
# https://medium.com/@susanne.schmid/image-normalization-in-medical-imaging-f586c8526bd1
# Percentile Normalization normalizes to a specific percentile e.g. 2% as lower bound 
# and 98% as upper bound. By taking the percentile, outliers are removed from the normalization. 
# This technique is interesting if the data has outliers and the distribution within the data varies.
# 2. Converts np image to 8-bit
# 3. Resize image
def process_np_img_slice(np_arr, perc_min, perc_max, resize_x=0, resize_y=0):
    # Norm np Image to 0-1
    np_arr = norm_by(np_arr, perc_min, perc_max)
    # fcn.get_nparr_info(np_arr)
    # Convert np image to 8-bit
    np_arr = convert_to_8bit(np_arr)
    # fcn.get_nparr_info(np_arr)
    # Resize image
    if(resize_x != 0 and resize_y != 0):
        np_arr = resize_np_img(np_arr, resize_x, resize_y)
    # fcn.get_nparr_info(img)
    return np_arr

# Get tile size and origin of a specific tile (M) from a specific channel (C)
# Tile index is not a 2D matrix, tiles are numbered from 1-end
# Get info about ALL tiles: img_data.get_all_mosaic_tile_bounding_boxes()
# Returns a (x, y) tuple
def get_tile_origin(czi_data, channel, tile_index, num_tiles_x, num_tiles_y):
    # Get offset for the left corner (x, y) of the first tile (M=0)
    # I do not know why the origin of the first tile it is not (0, 0)?
    # But this corrects for it
    tile0_data = czi_data.get_mosaic_tile_bounding_box(M=0, C=channel)
    # Read actual tile data
    tile_data = czi_data.get_mosaic_tile_bounding_box(M=tile_index, C=channel)  
    return((tile_data.x - tile0_data.x), (tile_data.y - tile0_data.y))

# Converts the mosaic tile index used here (x, y pos) 
# to the one which is used internally for czi files (1-end)
# and the other way around
# https://stackoverflow.com/questions/41433202/converting-the-index-of-1d-array-into-2d-array
def convert_index_1d_to_2d(index_1d, num_cols):
    col = index_1d % num_cols
    row = index_1d // num_cols
    return (row, col)
def convert_index_2d_to_1d(index_2d_row, index_2d_col, num_cols):
    return index_2d_col + (index_2d_row * num_cols)  

# Function reads the dimensions (x, y) of a tile in a czi mosaic
def read_tile_size(czi_data):
     # Image shape, e.g. [{'X': (0, 2752), 'Y': (0, 2208), 'C': (0, 1), 'M': (0, 336), 'S': (0, 1)}]
    shape = czi_data.get_dims_shape()
    return {'x': int(shape[0]['X'][1]), 'y': int(shape[0]['Y'][1])}

# Function reads the number of tiles in a czi mosaic
def read_tile_number(czi_data):
     # Image shape, e.g. [{'X': (0, 2752), 'Y': (0, 2208), 'C': (0, 1), 'M': (0, 336), 'S': (0, 1)}]
    shape = czi_data.get_dims_shape()
    # Check if czi_data is a mosaic file
    if(czi_data.is_mosaic()):
        return shape[0]['M'][1]
    else:
        return 0

# Function calculates the origin for each 4x4 slice in each tile
def get_slice_origin(czi_data,
                     channel,
                     tile_x, 
                     tile_y, 
                     slice_size_x, 
                     slice_size_y, 
                     num_tiles_x):
    # Read tile size: tuple (x, y)
    tile_size = read_tile_size(czi_data)
    # Convert 2d index to 1d index
    index = convert_index_2d_to_1d(tile_x, tile_y, num_tiles_x)
    # Read tile origin
    tile_origin_x, tile_origin_y = get_tile_origin(czi_data, channel, index, tile_x, tile_y)
    # print(f"Tile {tile_x}_{tile_y}: tile ori x: {tile_origin_x}, y: {tile_origin_y}")
    # Calculate offset to center the slice in the tile  
    x_offset = (tile_size['x'] - (2 * slice_size_x)) // 2
    y_offset = (tile_size['y'] - (2 * slice_size_y)) // 2
    # print(f"Tile {tile_x}_{tile_y}: offset x: {x_offset}, y: {y_offset}")
    # Calculate slice origin
    slice_origin_x = tile_origin_x + x_offset
    slice_origin_y = tile_origin_y + y_offset
    # Print
    # print(f"Tile {tile_x}_{tile_y}: tile ori x: {tile_origin_x}, y: {tile_origin_y} - slice ori x: {slice_origin_x}, y: {slice_origin_y}")
    return (slice_origin_x, slice_origin_y)

# Function slices a czi tile in four images (850x850px)
# start_x, start_y: slice origin = left upper corner of image[0][0]
# tile_x, tile_y: position in the mosaic tile array 
# The four images are saved as a png file
def slice_tile_x4(np_arr, 
                  img_name, 
                  channel, 
                  tile_x, 
                  tile_y, 
                  ori_x, 
                  ori_y, 
                  size_x, 
                  size_y, 
                  perc_min, 
                  perc_max, 
                  resize_x, 
                  resize_y, 
                  export_pth,
                  save_imgs=True):
    num_rows = 2
    num_cols = 2
    row_id = 0
    col_id = 0
    start_y = ori_y
    # Iterate over rows
    for i in range(num_rows):  
        start_x = ori_x     
        # Iterate over columns
        for j in range(num_cols):
            image_name = f'{img_name}_{tile_x}_{tile_y}_{i}_{j}.png'
            # print(image_name)
            # print(f'start_x: {start_x}, start_y: {start_y}')
            # Slice np image
            np_slice = slice_np_img(np_arr, channel, start_x, start_y, size_x, size_y)
            # get_nparr_info(np_slice)
            # Process slice (normalize, convert to 8-bit, resize)
            # Returns a numpy array
            np_slice = process_np_img_slice(np_slice, perc_min, perc_max, resize_x, resize_y)
            # get_nparr_info(np_slice)
            # Save image
            if(save_imgs):
                save_np_img(np_slice, export_pth, image_name)
            start_x += size_x  
            col_id += 1 
        start_y += size_y
        row_id += 1
    return True

# Function slices an image out of a czi tile (1700x1700px)
# start_x, start_y: slice origin = left upper corner of image[0][0]
# tile_x, tile_y: position in the mosaic tile array 
# The four images are saved as a png file
def slice_tile_x1(np_arr, 
                  img_name, 
                  channel, 
                  tile_x, 
                  tile_y, 
                  ori_x, 
                  ori_y, 
                  size_x, 
                  size_y, 
                  perc_min, 
                  perc_max, 
                  resize_x, 
                  resize_y, 
                  export_pth,
                  save_imgs=True):
    # Define image name
    image_name = f'{img_name}_{tile_x}_{tile_y}.png'
    # print(image_name)
    # print(f'start_x: {start_x}, start_y: {start_y}')
    # Slice np image
    np_slice = slice_np_img(np_arr, channel, ori_x, ori_y, size_x*2, size_y*2)
    # get_nparr_info(np_slice)
    # Process slice (normalize, convert to 8-bit, resize)
    # Returns a numpy array
    np_slice = process_np_img_slice(np_slice, perc_min, perc_max, resize_x, resize_y)
    # get_nparr_info(np_slice)
    # Save image
    if(save_imgs):
        save_np_img(np_slice, export_pth, image_name)
    return True

# Returns a list of all czi files in the czi folder
# without path and extension
def get_czi_file_list(czi_pth, czi_ext):
    # Get a list with all czi files in czi/ folder
    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    czi_list = [f for f in glob.glob(czi_pth + "*" + czi_ext)]
    # print(czi_list)
    czi_list_size = len(czi_list)
    # print(czi_list_size)
    # Remove extension
    for index in range(czi_list_size):
        # https://stackoverflow.com/questions/42798967/how-to-subtract-strings-in-python
        # Remove path from string
        if czi_list[index].startswith(czi_pth):
            temp_str = czi_list[index].replace(czi_pth, '')
        # Remove extension fron string
        if temp_str.endswith(czi_ext):
            temp_str = temp_str.replace(czi_ext, '')
        # Repalce list entry with filename only
        czi_list[index] = temp_str
    # Sort list
    czi_list.sort()
    return czi_list

# Function creates folder structure for each image in the czi folder
# Main folder: czi image name
# Subfolder: 4x/, 1x/, and full
def create_export_folder(exp_path, file_name):
    os.makedirs(exp_path + file_name)
    return True

# Function exports the full mosaic image at lower size (25%)
# Export as 8-bit grayscale image in .png format
def export_full_img(czi_data, 
                    file_name, 
                    channel, 
                    scale, 
                    perc_min, 
                    perc_max, 
                    exp_path, 
                    exp_full_path, 
                    save_img=True):
    # Create folder for export format
    os.makedirs(exp_path + file_name + "/" + exp_full_path)
    # Reads czi mosaic file for slicing (= full scale) and returns it as a numpy array
    img = read_czi_mosaic(czi_data, channel, scale)
    # fcn.get_nparr_info(img)
    image_name_full = f'{file_name}_full.png'
    img_full = process_np_img_slice(img, perc_min, perc_max, resize_x=0, resize_y=0)
    if(save_img): 
        exp_path =  exp_path + file_name + "/" + exp_full_path
        save_np_img(img_full, exp_path, image_name_full)
    # Free memory
    del img    
    return True

# Function exports the 4x slices form each tile of the mosaic image
# Export as 8-bit grayscale image in .png format
# slice format is '4x' or '1x'
def export_sliced_img(slice_format,
                      czi_data, 
                      file_name, 
                      channel,
                      slice_size_x,
                      slice_size_y,
                      num_tiles_x,
                      num_tiles_y,
                      resize_x,
                      resize_y, 
                      scale, 
                      perc_min, 
                      perc_max, 
                      exp_path,
                      exp_1x_path, 
                      exp_4x_path, 
                      save_img=True):  
            
    # Reads czi mosaic file for slicing (= full scale) and returns it as a numpy array
    img = read_czi_mosaic(czi_data, channel, scale)
    # fcn.get_nparr_info(img)
    
    # Set export path and image name prefix for 1x and 4x slices
    # Create folder for export format 
    if(slice_format == '1x'):
        exp_1x_path_full =  exp_path + file_name + "/" + exp_1x_path
        image_prefix_1x = f'{file_name}_1x'
        os.makedirs(exp_1x_path_full) 
    if(slice_format == '4x'): 
        exp_4x_path_full =  exp_path + file_name + "/" + exp_4x_path
        image_prefix_4x = f'{file_name}_4x'
        os.makedirs(exp_4x_path_full)  
    
    # Iterate over tile rows
    for i in range(num_tiles_y):    
        # Iterate over tile columns
        for j in range(num_tiles_x):
            # Get slice origin for specific tile
            (slice_origin_x, slice_origin_y) = get_slice_origin(czi_data,
                                                                channel,
                                                                i, 
                                                                j, 
                                                                slice_size_x, 
                                                                slice_size_y, 
                                                                num_tiles_x) 
            
            # 1x slices (max size)
            if(slice_format == '1x'):            
                # Slice the tile and save the 1x images
                slice_tile_x1(img, 
                              image_prefix_1x, 
                              channel, 
                              i, 
                              j, 
                              slice_origin_x, 
                              slice_origin_y, 
                              slice_size_x, 
                              slice_size_y, 
                              perc_min, 
                              perc_max, 
                              resize_x, 
                              resize_y, 
                              exp_1x_path_full,
                              save_img)
               
            # 4x slices (max size/4)  
            if(slice_format == '4x'):       
                # Slice the tile and save the 4x images
                slice_tile_x4(img, 
                              image_prefix_4x, 
                              channel, 
                              i, 
                              j, 
                              slice_origin_x, 
                              slice_origin_y, 
                              slice_size_x, 
                              slice_size_y, 
                              perc_min, 
                              perc_max, 
                              resize_x, 
                              resize_y, 
                              exp_4x_path_full,
                              save_img)
                
    # Free memory
    del img
    return True

"""
# Function calculates the origin for each 4x4 slice in each tile <- Old version
def get_slice_origin(tile_x, 
                     tile_y, 
                     slice_size_x, 
                     slice_size_y, 
                     tile_size_x, 
                     tile_size_y,
                     overlap_perc):
    # Calculate overlap of mosaic images in px for x and y
    overlap_x = int(tile_size_x * (overlap_perc / 100))
    overlap_y = int(tile_size_y * (overlap_perc / 100))
    # print(f"Tile {tile_x}_{tile_y}: overlap x: {overlap_x}, y: {overlap_y}")
    # Calculate tile origin
    tile_origin_x = tile_y * (tile_size_x - overlap_x)
    tile_origin_y = tile_x * (tile_size_y - overlap_y)
    # print(f"Tile {tile_x}_{tile_y}: tile ori x: {tile_origin_x}, y: {tile_origin_y}")
    # Calculate offset
    x_offset = (tile_size_x - (2 * slice_size_x)) // 2
    y_offset = (tile_size_y - (2 * slice_size_y)) // 2
    # print(f"Tile {tile_x}_{tile_y}: offset x: {x_offset}, y: {y_offset}")
    # Calculate slice origin
    slice_origin_x = tile_origin_x + x_offset
    slice_origin_y = tile_origin_y + y_offset
    # Print
    # print(f"Tile {tile_x}_{tile_y}: tile ori x: {tile_origin_x}, y: {tile_origin_y} - slice ori x: {slice_origin_x}, y: {slice_origin_y}")
    return (slice_origin_x, slice_origin_y)
"""