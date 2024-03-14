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

# import aicspylibczi
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os

# Get image information
def get_czi_info(czi_data):
    # Image type
    print("Type:", type(czi_data))
    # Image dimensions, e.g.'SCMYX'
    print("Dimensions:", czi_data.dims)
    # Image size, e.g. (1, 3, 130, 2208, 2752) (time?, channel, no tiles, height, width)
    print("Size:", czi_data.size)
    # Image shape, e.g. [{'X': (0, 2752), 'Y': (0, 2208), 'C': (0, 3), 'M': (0, 130), 'S': (0, 1)}]
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
def resize_np_img(np_arr, new_size):
    pil_img = convert_np_to_pil(np_arr)
    pil_resize = pil_img.resize((new_size['x'], new_size['y']))
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
def process_np_img_slice(np_arr, perc_norm):
    # Norm np Image to 0-1
    np_arr = norm_by(np_arr, perc_norm['min'], perc_norm['max'])
    # fcn.get_nparr_info(np_arr)
    # Convert np image to 8-bit
    np_arr = convert_to_8bit(np_arr)
    # fcn.get_nparr_info(np_arr)
    return np_arr

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
    
# Function reads the number of channels of a tile in a czi mosaic
def read_channel_num(czi_data):
     # Image shape, e.g. [{'X': (0, 2752), 'Y': (0, 2208), 'C': (0, 1), 'M': (0, 336), 'S': (0, 1)}]
    shape = czi_data.get_dims_shape()
    return int(shape[0]['C'][1])

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
    print(f"Folder {exp_path + file_name + '/'} for exported images created.")
    return True

# Get origin (upper left corner) of a specific tile
# Tile index is not a 2D matrix, tiles are numbered from 1-end
# Get info about ALL tiles: img_data.get_all_mosaic_tile_bounding_boxes()
# Returns a (x, y) tuple
def get_tile_origin(czi_data, tile_index):
    # Get offset for the left corner (x, y) of the first tile (M=0)
    # I do not know why the origin of the first tile it is not (0, 0), but this corrects for it
    # This is done only for channel 0 as images of all channels are located in the exact same position
    channel = 0
    tile0_data = czi_data.get_mosaic_tile_bounding_box(M=0, C=channel)
    # Read actual tile data
    tile_data = czi_data.get_mosaic_tile_bounding_box(M=tile_index, C=channel)  
    return((tile_data.x - tile0_data.x), (tile_data.y - tile0_data.y))

# Function calculates the origin for each 4x4 slice in each tile
def get_slice_origin(czi_data,
                     tile_pos_x,
                     tile_pos_y,
                     slice_size, 
                     num_tiles_x):
    # Read tile size: tuple (x, y)
    tile_size = read_tile_size(czi_data)
    # Convert 2d index to 1d index
    tile_index = convert_index_2d_to_1d(tile_pos_x, tile_pos_y, num_tiles_x)
    # Read tile origin
    tile_origin_x, tile_origin_y = get_tile_origin(czi_data, tile_index)
    # print(f"Tile {tile_x}_{tile_y}: tile ori x: {tile_origin_x}, y: {tile_origin_y}")
    # Calculate offset to center the slice in the tile  
    x_offset = (tile_size['x'] - (slice_size['x'])) // 2
    y_offset = (tile_size['y'] - (slice_size['y'])) // 2
    # print(f"Tile {tile_x}_{tile_y}: offset x: {x_offset}, y: {y_offset}")
    # Calculate slice origin
    slice_origin_x = tile_origin_x + x_offset
    slice_origin_y = tile_origin_y + y_offset
    # Print
    # print(f"Tile {tile_x}_{tile_y}: tile ori x: {tile_origin_x}, y: {tile_origin_y} - slice ori x: {slice_origin_x}, y: {slice_origin_y}")
    return {'x': slice_origin_x, 'y': slice_origin_y}

# Slice np image
# def slice_np_img(channel, np_arr, start_x, start_y, size_x, size_y):
def slice_np_img(np_arr, origin, size):
    slice_end = {'x': (origin['x'] + size['x']), 'y': (origin['y'] + size['y'])}
    return np_arr[origin['y']:slice_end['y'], origin['x']:slice_end['x']]

# Function exports the full mosaic image at lower size (25%)
# Export as 8-bit rgb image with 3 channels in .png format
def export_full_img(czi_data, 
                    export_type,
                    file_name,
                    num_channels, 
                    scale, 
                    perc_norm, 
                    exp_path, 
                    exp_type_path, 
                    save_img=True):
    
    if(export_type['full']):

        print(f"Export of full image at {scale*100}% size is starting. Please wait...")

        # Create folder for export format
        if(save_img):
            os.makedirs(exp_path + file_name + "/" + exp_type_path['full'])
        # Set image name
        image_name_full = f'{file_name}_full.png'

        # Create list with all channels
        img_channels = []
        # Iterate over channels
        for channel in range(num_channels):
            # Reads one channel of the czi mosaic file for slicing (= full scale) and returns it as a numpy array
            img = read_czi_mosaic(czi_data, channel, scale)
            # fcn.get_nparr_info(img)
            # delete dimension 0 (e.g. shape of (1, 7176, 6880))
            img = img[0, :, :] 
            # Process slice
            img = process_np_img_slice(img, perc_norm[channel])    
            # Append channel to list
            img_channels.append(img)
            # Free memory
            del img 

        # Combine channels to one RGB image
        rgb_img = np.dstack(tuple(img_channels))
        pil_img = Image.fromarray(rgb_img, mode='RGB')
        # Save image
        if(save_img): 
            exp_path =  exp_path + file_name + "/" + exp_type_path['full']
            pil_img.save(exp_path + image_name_full)

        print(f"Export of full image finished.")

    return True

# Function slices images of different sizes out of a czi tile
# The images are saved as rgb .png file
def slice_tiles(czi_data,
                exp_type,
                file_name, 
                num_channels, 
                num_tiles,
                slice_size, 
                perc_norm,
                resize,
                import_scale,
                exp_path,
                exp_type_path,
                save_img=True):
    
    print(f"Prepare image for slicing. Please wait...")
    # Create list with all channels of the mosaic file
    img_channels = []
    # Iterate over channels
    for channel in range(num_channels):
        # Reads one channel of the czi mosaic file for slicing and returns it as a numpy array
        img = read_czi_mosaic(czi_data, channel, import_scale)
        # get_nparr_info(img)
        # Images are of shape (1, 28704, 27520) where 1 seems to be the channel
        # First dimension needs to be deleted as the export can only occure channel wise
        # I couldn't manage to export a numpy array with 3 channel dimensions
        img = img[0, :, :]
        # Process slice
        img = process_np_img_slice(img, perc_norm[channel])   
        # Append channel to list
        img_channels.append(img)
        # Free memory
        del img 
    print(f"Image is ready for slicing.")
    
    # Iterate over export types
    for dict_key in exp_type:

        # If the export type is set to true, images are sliced and saved
        if(exp_type[dict_key] and dict_key != 'full'):
            print(f"Export of {dict_key} image slices is starting. Please wait...")

            # Set directories for sliced images and image prefix
            exp_path_full =  exp_path + file_name + "/" + exp_type_path[dict_key]
            if(save_img):
                os.makedirs(exp_path_full) 
            image_prefix = f'{file_name}_{dict_key}'

            # Set number of cols/rows for further slicing the 1x slice
            if(dict_key == '1x'):
                num_rows = 1
                num_cols = 1
            elif(dict_key == '4x'):
                num_rows = 2
                num_cols = 2   
            elif(dict_key == '16x'):          
                num_rows = 4
                num_cols = 4   

            # Iterate over mosaic tiles
            for i in range(num_tiles['y']):   
                for j in range(num_tiles['x']):

                    # Get slice origin for 1x tile
                    slice_origin = get_slice_origin(czi_data, i, j, slice_size['1x'], num_tiles['x']) 
                    # print(f"\t> Processing {dict_key} image slice {i}_{j}...")

                    row_id = 0
                    col_id = 0
                    start_y = slice_origin['y']
                    # Iterate over rows
                    for k in range(num_rows):  
                        start_x = slice_origin['x']     
                        # Iterate over columns
                        for l in range(num_cols):
                            # Define image name
                            image_name = f'{image_prefix}_{i}_{j}_{k}_{l}.png'
                            # print(image_name)
                            # Iterate over all channels and slice images
                            img_slices = []
                            for n in range(num_channels):
                                # Create slice coordinates
                                origin = {'x': start_x, 'y': start_y}
                                # Slice np image
                                sliced_ch = slice_np_img(img_channels[n], origin, slice_size[dict_key])
                                # get_nparr_info(np_slice)
                                # Resize slice
                                sliced_ch = resize_np_img(sliced_ch, resize)
                                # get_nparr_info(np_slice)
                                # Append slice to channel list
                                img_slices.append(sliced_ch)
                            # Combine channels to one RGB image
                            rgb_slice = np.dstack(tuple(img_slices))
                            pil_img = Image.fromarray(rgb_slice, mode='RGB')
                            # Save image
                            if(save_img): 
                                pil_img.save(exp_path_full + image_name)
                            # Move slice origin forward
                            start_x += slice_size[dict_key]['x']  
                            col_id += 1 
                        start_y += slice_size[dict_key]['y'] 
                        row_id += 1

            print(f"Export of {dict_key} image slices finished.")           

    return True

# Split an Image in XxY pices
# From https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
# Not used here as crpos must be taken from the original full image and then resized
def imgcrop(input, xPieces, yPieces):
    filename, file_extension = os.path.splitext(input)
    im = Image.open(input)
    imgwidth, imgheight = im.size
    height = imgheight // yPieces
    width = imgwidth // xPieces
    for i in range(0, yPieces):
        for j in range(0, xPieces):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            a = im.crop(box)
            try:
                a.save("export/" + filename + "-" + str(i) + "-" + str(j) + file_extension)
            except:
                pass

