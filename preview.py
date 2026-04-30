import nibabel as nib
import numpy as np

# Replace 'your_file.nii.gz' with the path to your file
try:
    nifti_file = nib.load('/home/rmedu-04/SSLP/totalsegmentator/s0000/ct.nii.gz')
    print("NIfTI file loaded successfully.")
except FileNotFoundError:
    print("Error: The specified file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Get the image data as a NumPy array
if 'nifti_file' in locals():
    image_data = nifti_file.get_fdata()
    print("Image data shape:", image_data.shape)
    print("Image data type:", image_data.dtype)


if 'nifti_file' in locals():
    # Access the header
    header = nifti_file.header
    
    # Get voxel sizes
    voxel_dims = header.get_zooms()
    print("Voxel dimensions (x, y, z):", voxel_dims[:3])

    # Get the data type
    data_type = header.get_data_dtype()
    print("Data type:", data_type)

import matplotlib.pyplot as plt

if 'image_data' in locals() and image_data.ndim >= 3:
    # Get the middle slice along the z-axis
    middle_slice_index = image_data.shape[2] // 2
    middle_slice = image_data[:, :, middle_slice_index]

    # Display the slice
    plt.imshow(middle_slice.T, cmap='gray', origin='lower')
    plt.title(f"Slice {middle_slice_index}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.colorbar(label='Intensity')
    plt.show()
