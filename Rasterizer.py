#################### 
# @Author: Tanmay.Mukherjee  
# @Date: 2025-01-03 22:53:55  
# @Last Modified by: Tanmay.Mukherjee  
# @Last Modified time: 2025-01-03 22:53:55 
####################
import numpy as np
import pyvista as pv
import vtk
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import cv2


class Raster_Setup(object):
    
    def __init__(self, target_resolution = (256,256), source_normal = [0,1,0], target_normal = [0,0,1], scalar_to_plot = 'uy',
                 min_scale = -1, max_scale = 1, scale_val = 255, no_of_planes = [3,0], position = np.array([0,0,0]),
                 target_size = 1.25, slice_thickness = 0):
        
        self.target_res = target_resolution
        self.source_normal = source_normal # [0,1,0] for SA, [1,0,1] for LA  
        self.target_normal = target_normal # Most likely [0,0,1]
        self.scalar_to_plot = scalar_to_plot
        self.temp_normal = source_normal
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_val = scale_val
        self.no_of_planes = no_of_planes # [no of SA planes, no of LA_planes]
        self.position = position
        self.target_size = target_size
        self.slice_thickness = slice_thickness

    

    def scale_scalars_by_color_value(self, vector = np.array([0]), scale_by_val = None):

        if scale_by_val is None:
            return (self.scale_val*(vector - self.min_scale)/ (self.max_scale - self.min_scale)).astype(np.uint8)
        else:
            return scale_by_val*(vector - self.min_scale)/ (self.max_scale - self.min_scale)

    
    def initialize_raster_grid(self, i_size = 15, j_size = 15, center = [0,0,0]):

        if self.slice_thickness == 0:
            self.raster_grid = pv.Plane(direction = self.temp_normal, center = center, i_size = self.target_size, j_size = self.target_size, \
                        i_resolution= self.target_res[0], j_resolution= self.target_res[1])
        else:

            roi_plane = pv.Plane(direction=self.temp_normal, center=center, 
                                i_size = self.target_size, j_size = self.target_size, i_resolution= self.target_res[0], j_resolution= self.target_res[1])

            extrude = vtk.vtkLinearExtrusionFilter()
            extrude.SetInputData(roi_plane)
            extrude.SetScaleFactor(self.slice_thickness)
            # extrude.SetVector([1,0,0])
            extrude.Update()
            self.raster_grid = pv.wrap(extrude.GetOutput())

        self.raster_grid_cell_centers = self.raster_grid.cell_centers()

        # return raster_grid, raster_grid_cell_centers

    def map_mesh_vertices_to_raster_cells(self, object_location = None, object_connectivity = None, object_scalars = None):

        ## Comparing pointwise location of object and grid cell vertices 
        grid_points = self.raster_grid_cell_centers.points
        
        idx_l = []

        ## Looping through the points in the object to calculate closest faces in grid
        for j, point in enumerate(object_location):
            # print(point[1])
            # Calculate distances to all grid points for all faces
            dist_stacks = []
            distances = np.sqrt((grid_points[:, 0] - point[0])**2 +
                                    (grid_points[:, 1] - point[1])**2 +
                                    (grid_points[:, -1] - point[-1])**2)
            
            min_dist_idx = np.argmin(distances)

            dist_stacks.append((distances[min_dist_idx], min_dist_idx))
            closest_face = min(dist_stacks, key=lambda x: x[0])

            idx_l.append(np.hstack((closest_face[1], object_connectivity[j], object_scalars[j,0])))
            
        ## Assigning random pixel intensities to each cell
        grid_cells = np.vstack(idx_l)

        return grid_cells

    def paint_by_cell_connectivity(self, raster_grid_cells = None, img = None):

        ## Finding connectivity of the pixels in raster grid
        unique_rows, indices = np.unique(raster_grid_cells[:, 1:], axis=0, return_inverse=True) 

        # Initialize a list to store the grouped elements from column 0
        grouped_plx_connectivity = []

        # Group the grid cells based on their connectivity
        for row in range(len(unique_rows)):
            rows, cols = np.unravel_index(raster_grid_cells[indices == row, 0], img.shape)
            idx_array = np.hstack((np.c_[rows],np.c_[cols]))
            grouped_plx_connectivity.append(idx_array)

        for idx, cnt in enumerate(grouped_plx_connectivity):

            color_val = int(unique_rows[idx,-1])

            cv2.drawContours(img, [cnt], 0, color_val ,-1)
        
        return img
    
    def calculate_sdf(self, plx = pv.PolyData, mesh = pv.PolyData):
        
        image=plx['mask'].reshape(-1,self.target_res[0])
        edt = distance_transform_edt(image) 
        signed_distance_field = np.where(image, -edt, edt)


        return signed_distance_field
        

    
    def __getattribute__(self, name):
        return object.__getattribute__(self, name)
    
    def set_attribute(self, attr_name, value):
        setattr(self, attr_name, value)  # Dynamically set an attribute


class Kernel():

    def __init__(self, size = 5, frequency = 10, std = 2, phase = 1):

        self.size = size
        self.frequency = frequency
        self.std = std
        self.phase = phase

    def gauss_kernel(self):
        
        kernel = np.zeros((self.size, self.size))
        center = self.size // 2

    
        for i in range(self.size):
            for j in range(self.size):
                kernel[i, j] = (1/(2*np.pi*self.std**2)) * np.exp(-((i - center)**2 + (j - center)**2) / (2*self.std**2))
    
        return kernel / np.sum(kernel)
    
    def convolve_2d(self, img, kernel):
        return cv2.filter2D(img, -1, kernel)
    
    def add_noise(self, img, mean = 25, std = 10):
        # Generate Gaussian noise
        gaussian_noise = np.random.normal(mean, std, img.shape)

        # Add noise to the image array
        noisy_image_array = img + gaussian_noise
        
        # noisy_image_array = image_array.copy() 

        # Clip values to ensure they stay within the valid range [0, 255]
        noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)

        return noisy_image_array
        
