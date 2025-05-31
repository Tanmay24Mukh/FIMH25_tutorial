#################### 
# @Author: Tanmay.Mukherjee  
# @Date: 2025-01-04 11:12:37  
# @Last Modified by: Tanmay.Mukherjee  
# @Last Modified time: 2025-01-04 11:12:37 
####################

import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
import argparse
from tqdm import tqdm
import multiprocessing as mp
import pyvista as pv

from Rasterizer import Raster_Setup, Kernel
from Mesh import Mesh3D
from DataIO import Read_Data, Write_Data
from Methods import calculate_affine_transformation_matrix

def create_images(file = 'mesh.vtk', rasterizer = Raster_Setup, kernel = Kernel):

    """
    Creates images from mesh.vtk files. Images are initialized as grids (pv.Plane)
    and raycasting is implemented to rasterize 2D slices, determined via input origin 
    and normal. 
    """
    
    Mesh = Mesh3D(mesh= Read_Data.read_mesh(fid = file))
    Mesh.center_and_normalize_mesh()
    Mesh.covert_point_data_to_cell_data()
    xmin, xmax, ymin, ymax, zmin, zmax = Mesh.mesh.bounds

    x_sample, y_sample, z_sample = np.linspace(xmin,xmax,rasterizer.no_of_planes + 2)[1:-1], \
                                 np.linspace(ymin,ymax,rasterizer.no_of_planes + 2)[1:-1], \
                                 np.linspace(zmin,zmax,rasterizer.no_of_planes + 2)[1:-1]
    
    sample_xyz = np.array([[vv, y_sample[ii], z_sample[ii]] for ii,vv in enumerate(x_sample)])

    # print('$$$$$$$$$$$$$$')

    results = []
    for normal in rasterizer.source_normal:
        logic_normal = np.array(normal != 0)
        origins = logic_normal*normal*np.array(sample_xyz)
        rasterizer.temp_normal = normal

        # plt.figure()
        for idx, origin in enumerate(origins):
            I, mask, affine_matrix = vtk_slice_to_np_img(Mesh=Mesh, rasterizer=rasterizer, kernel=kernel, origin=origin, normal=normal)
            
            target_normal = np.array(rasterizer.target_normal, dtype=float)
            og_normal = np.array(normal, dtype=float)
            affine_T_matrix = [list(row) for row in np.array(affine_matrix, dtype = float)]
            # print(affine_T_matrix)
            specs = {'plane number': str(idx),
                     'original_normal': list(og_normal),
                     'proj_normal': list(target_normal),
                     'height': int(rasterizer.target_res[0]),
                     'width': int(rasterizer.target_res[1]),
                     'w_size': float(rasterizer.target_size), 
                     'h_size': float(rasterizer.target_size), 
                     'affine_matrix': affine_T_matrix
                     }
            results.append((I, mask, specs))

            # plt.figure()
            # plt.imshow(I, cmap = 'gray')
            # plt.show()



    return results    

def vtk_slice_to_np_img(Mesh = Mesh3D, rasterizer = Raster_Setup, kernel = Kernel, origin = [0,0,0], normal = [0,0,1]):


    Mesh.extract_2d_slice(origin=origin, normal=normal, thickness=rasterizer.slice_thickness)
    # Mesh.extract_2d_slice(origin=origin, normal=normal, thickness=0.2)

    # Mesh.extract_slice_with_thickness(origin=Mesh.sl.center, normal=normal)

    # Mesh.sblock.plot()
    # Mesh.sl.plot()

    rasterizer.initialize_raster_grid(center=Mesh.sl.center)

    
    if len([cell for cell in Mesh.sl.cell]) > 0:

        vertices_stack, faces_stack, scalars_stack = Mesh.extract_slice_vertex_and_face_data()

        # print('Min scalars is: ', np.min(scalars_stack))

        scaled_scalars_stack = rasterizer.scale_val + np.zeros((scalars_stack.shape[0],1)).astype(np.uint8)

        # print('Min scalars is: ', np.min(scaled_scalars_stack))

        ## List containing connectivity data for the rasterized grid
        sl_2_raster = rasterizer.map_mesh_vertices_to_raster_cells(object_location = vertices_stack, 
                    object_connectivity = faces_stack, object_scalars = scaled_scalars_stack)
        
        ## Initializing numpy image array
        I_array = np.zeros((rasterizer.raster_grid_cell_centers.points.shape[0],1))
        # print("I_array shape is: ", I_array.shape)
        I_initial = I_array.reshape((rasterizer.target_res[0],rasterizer.target_res[1]))

        ## Painting numpy image by cell connectivity data
        I_mask = rasterizer.paint_by_cell_connectivity(raster_grid_cells=sl_2_raster, img=I_initial)
        
        if kernel is None:
            I = I_mask.copy()
        else: 
            I_conv = kernel.convolve_2d(img=I_mask, kernel=kernel.gauss_kernel())
            I_noise = kernel.add_noise(img=I_conv, mean=5, std= 1)
            I = I_conv + I_noise

        mask = I_mask.copy()


    else:

        I =  np.zeros((rasterizer.target_res[0], rasterizer.target_res[1]))
        mask = I.copy()



    affine_matrix = calculate_affine_transformation_matrix(original_plane_normal=normal, 
                                                            target_plane_normal=np.array([0,0,1]),
                                                            translation_vector=rasterizer.raster_grid.center)
    
    rasterizer.raster_grid['intensity'] = I.T.reshape(-1)
    # rasterizer.raster_grid['intensity'] = I.reshape(-1)
    # rasterizer.raster_grid.plot()

    return I, mask, affine_matrix


def img_to_xyz_coords(img = np.array([0]), specs = {}):


    i_res, j_res = specs['h_size'] ,specs['w_size']

    affine_matrix = np.array(specs['affine_matrix'])

    # print(affine_matrix)


    img_width_coords = np.linspace(-j_res/2,j_res/2,img.shape[1])
    img_height_coords = np.linspace(-i_res/2,i_res/2,img.shape[0])

    [img_H, img_W] = np.meshgrid(img_height_coords, img_width_coords)

    x_coords, y_coords  = img_W.reshape(-1), img_H.reshape(-1)
    z_coords = np.zeros((x_coords.shape[0],1))
    
    
    coords = np.hstack((np.c_[x_coords],np.c_[y_coords], np.c_[z_coords]))
    rotated_coords = np.dot(coords, affine_matrix[:3,:3].T)

    rotated_and_translated_coords = rotated_coords + affine_matrix[:3,3]    

    return rotated_and_translated_coords
