# READ_CZI

The program exports images in different sizes from a microscopic .czi mosaic file (Zeiss format).
The images can be used used to train neural networks.

Export sizes:
The .czi file MUST have 3 channels (for now) and exports different slice sizes:
- full: a full overview image in 25% of the original size
- 1x: one slice per mosaic tile (2048x2048px from the original image, resized to 512x512px)
- 4x: 1x slice further sliced in 4 images (1024x1024px from the original image, resized to 512x512px)
- 16x: 1x slice further sliced in 16 images (512x512px from the original image, no resizing)

Folder structure:
Original .czi files must be copied into the folder "czi/". Finished images are exported to folder "export/". 
The program automatically iterates over all .czi files in the czi folder and processes them. 
For every .czi file a new folder is created in the export folder, which is named after the .czi file.
When an export is done, the folder will contain 4 subfolders for different slice sizes (full/, 1x/, 4x/, 16x/).

Export format:
The exported images are all 512x512px in size and consist of the three original channels (rgb channels).
The export format is .png, the alpha channel is not used.

Microscope parameters:
Microscopic images are taken using a 60x oil objective. Mosaic images were taken with a motorized microscope stage
(x: 10 images, y: 13 images). This way a lot of training images can be created in a short time.
The size of one tile is 2752x2208 px. The cells are stained with two fluorescent dyes (Ch0 and 1), Ch2 is a
brightfield image using DIC contrast. Mosaic images must be taken without an overlap!

Libraries:
Based on library: aicspylibczi (https://github.com/AllenCellModeling/aicspylibczi)
Documentation: https://allencellmodeling.github.io/aicspylibczi/aicspylibczi.html

Image processing:
- Percentile normalization
- Conversion to 8-bit
- Downsampling to 512x512 (except full image)
- Expeort as 3 channel .png file


