#################### 
# @Author: Tanmay.Mukherjee  
# @Date: 2025-01-04 02:13:01  
# @Last Modified by: Tanmay.Mukherjee  
# @Last Modified time: 2025-01-04 02:13:01 
####################
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
import argparse
from tqdm import tqdm
import multiprocessing as mp

from Rasterizer import Raster_Setup, Kernel
from DataIO import Read_Data, Write_Data
from Processes import create_images

def create_input_images():

 
    arguments = determine_inputs()
    rasterizer = Raster_Setup()
    kernel = Kernel()
    

    phantom_path = "Datasets\\"
    study = arguments.study
    
    # Accessing mesh vtk files
    vtk_path = os.path.join(phantom_path, study) 
    vtk_files = Read_Data.read_vtk_files(path = vtk_path)
    time_points = (np.unique(np.linspace(arguments.T[0], arguments.T[1], arguments.T[2])) - 1).astype(int)
    vtk_files_ = np.array(vtk_files)[time_points]

    store_path = "Datsets\\"

    # Setting image and image specs file location
    # Creating image encoder input images path
    img_path = os.path.join(store_path, study, "images_SR")
    mask_path = os.path.join(store_path, study, "masks_SR")
    specs_path = os.path.join(store_path, study, "img_specs")


    # Initializing the mesh to image rasterizer class
    rasterizer.target_res = arguments.R
    rasterizer.source_normal = np.array(arguments.N).reshape(-1,3)
    rasterizer.scalar_to_plot = arguments.scalar  
    rasterizer.min_scale = arguments.scaling[0]
    rasterizer.max_scale = arguments.scaling[-1]
    rasterizer.scale_val = arguments.intensity
    rasterizer.no_of_planes = arguments.P
    rasterizer.target_size = 1.25
    
    write_pngs = True

    num_processes = mp.cpu_count()
    print('Multiprocessing on ' + str(num_processes) + ' cores')

    begin = time()
    # for i, file in vtk_files_:
    print('Creating images from mesh vtks\n')
    with mp.Pool(processes=num_processes) as pool:
        with tqdm(total=len(vtk_files_)) as pbar:
            results = pool.starmap(create_images, [(file, rasterizer, kernel) for i, file in vtk_files_])
            pbar.update(len(vtk_files_))

        I_l, mask_l, specs_l = [], [], []
        for file_results in results:

            I_temp = np.stack([I for I,_,_ in file_results], axis=0)
            I_l.append(I_temp)

            mask_temp = np.stack([mask for _,mask,_ in file_results], axis=0)
            mask_l.append(mask_temp)

            specs_temp = [specs for _, _, specs in file_results]
            specs_l.append(specs_temp)

    I_stack = np.stack(I_l, axis = 0) # timepoint, plane_no, height, width
    mask_stack = np.stack(mask_l, axis = 0) # timepoint, plane_no, height, width

    print('\n Writing images, masks, and specs\n')

    for tf in tqdm(range(I_stack.shape[0])):

        if write_pngs:
            for k, I in enumerate(I_stack[tf]):
                # print(I_stack[tf,k,:,:].shape)
                png_path = os.path.join(img_path, "pngs", "SA_" + str(k).zfill(2))
                if not os.path.isdir(png_path): os.makedirs(png_path)
                img_fid = study + '_img_t_' + str(tf).zfill(4) + '.png'
                Write_Data.write_png_images(img=I_stack[tf,k,:,:], path=png_path, fid=img_fid)
        else:


            img_fid = study + '_img_t_' + str(tf).zfill(4) + '.nii.gz'
            specs_fid = study + '_specs_t_' + str(tf).zfill(4) + '.json'
            mask_fid = study + '_mask_t_' + str(tf).zfill(4) + '.nii.gz'


            Write_Data.write_images(img = I_stack[tf,:,:,:], path=img_path, fid=img_fid)
            Write_Data.write_images(img = mask_stack[tf,:,:,:], path=mask_path, fid=mask_fid) 
            Write_Data.write_img_specs(specs=specs_l[tf], path=specs_path, fid=specs_fid)
        

    end = time()
    print('Total run time is: ', end-begin)


def determine_inputs():

    parser = argparse.ArgumentParser(
                    prog = 'Create_Inputs.py',
                    description = 'Creates images from vtks')
    
    parser.add_argument('-S', '--study', default= 'M202', type=str, help = 'name of study')
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



if __name__ == "__main__":

    create_input_images()

      
