# READ_CZI

Program reads a .czi microscopic mosaice file (Zeiss) with one channel and exports different slice sizes.
Original tile size with 60x oil objective: 2752x2208 px. So far an image with only one channel can be processed.
The images are used to train neural networks.

Based on library: aicspylibczi

Exported images:
- Full image: all tiles exported at 25% size
- 1x slices: slicing of a 1700x1700 px area in the center of each tile
- 4x slices: 1x slices are cut in 4 images of dimension 850x850 px

Image processing:
- Percentile normalization
- Conversion to 8-bit
- Downsampling to 512x512 (except full image)
- Expeort as .png file


