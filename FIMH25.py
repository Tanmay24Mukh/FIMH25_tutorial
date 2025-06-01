import os
import numpy as np
import pyvista as pv
import vtk
import matplotlib.pyplot as plt
from time import time
import argparse
from tqdm import tqdm
import multiprocessing as mp

from Rasterizer import Raster_Setup, Kernel
from DataIO import Read_Data, Write_Data
from Processes import create_images
from Mesh import Mesh3D

def main():

    ## Gather user-defined inputs
    inputs = parse_inputs()

    ## Initialize mesh to image rasterizer
    rasterizer = Raster_Setup()

    ## Initialize image kernel 
    kernel = Kernel()

    ## Access vtk mesh files
    # Read all vtk files in a directory
    input_path = inputs.path
    study_id = inputs.study
    vtk_mesh_path = os.path.join(input_path, study_id)
    vtk_mesh_files = Read_Data().read_files(path = vtk_mesh_path, ext = '.vtk')

    # Read specific indices to adjust target temporal resolution
    start, end, skip = inputs.T[0], inputs.T[1], inputs.T[2]
    
    if skip!=0:
        indices = np.arange(start,end)[::skip]
    else:
        indices = np.arange(start,end)

    vtk_mesh_files = np.array(vtk_mesh_files)[indices]

    # Set rasterization partmeters
    rasterizer.target_res = inputs.R # Target resolution of the image
    rasterizer.source_normal = np.array(inputs.N).reshape(-1,3) # Orientation of the 'scanner'
    rasterizer.scalar_to_plot = inputs.scalar  
    rasterizer.min_scale = inputs.scaling[0]
    rasterizer.max_scale = inputs.scaling[-1]
    rasterizer.scale_val = inputs.intensity # Target image intensity 
    rasterizer.no_of_planes = inputs.P
    rasterizer.target_size = 1.25
    rasterizer.slice_thickness = 0.
    
    # Create grayscale images from vtk meshes
    results = create_images(file = vtk_mesh_files[0][-1], rasterizer = rasterizer, kernel = kernel)

    plt.figure()
    plt.imshow(results[0][0], cmap = 'gray', clim = [0,255])
    plt.show()




def parse_inputs():

    parser = argparse.ArgumentParser(
                    prog = 'FIMH25.py',
                    description = 'Synthetic phantom generation')
    
    parser.add_argument( '--path', default= "Datasets\\" , type=str, help = 'input path')
    parser.add_argument('-S', '--study', default= 'Mouse_LV_FE', type=str, help = 'name of study')
    parser.add_argument('-T', nargs= '+', default= [1, 2, 1], type = int,  help = 't_start, t_end, no_of_divisions')
    parser.add_argument('-P', type = int, default=30, help= 'no_of_planes')
    parser.add_argument('-N', nargs='+', type=np.float16, default=[0,0,1], help = 'direction of the SA planes')
    parser.add_argument('-R', nargs='+', type = np.int32, default=[256,256], help='target resolution of images')
    parser.add_argument('-U', '--scalar', type = str, default='ux', help = 'scalar_to_plot ux, uy, uz, e11, e22, e33')
    parser.add_argument('-C', '--scaling', nargs='+', type = int,  default=[-1,1], help = 'min, max scaling for scalars')
    parser.add_argument('-I', '--intensity', type = int,  default= 255, help = '1 channel intensity value')
    parser.add_argument('--num_of_cores', type = int, default = 8, help = 'Number of processes (cores)')
    
    arguments = parser.parse_args()
    
    return arguments
if __name__ == '__main__':
    main()
    # extract_slice_with_thickness()