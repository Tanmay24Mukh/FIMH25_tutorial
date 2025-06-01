#################### 
# @Author: Tanmay.Mukherjee  
# @Date: 2025-01-03 22:26:41  
# @Last Modified by: Tanmay.Mukherjee  
# @Last Modified time: 2025-01-03 22:26:41 
####################

import numpy as np
import pyvista as pv
import vtk


class Mesh3D():

    def __init__(self, mesh):
        # if not isinstance(mesh, pv.UnstructuredGrid):
        #     raise ValueError("mesh must be a pv.UnstructuredGrid")
        self.mesh = mesh

    def extract_2d_slice(self, origin=[0, 0, 0], normal=[0, 0, 1], thickness = 0):
        if thickness == 0:
            self.sl = self.mesh.slice(origin=origin, normal=normal)
        else:
            slice = self.mesh.slice(origin=origin, normal=normal)
            roi_plane = pv.Plane(direction=normal, center=slice.center, 
                                i_size=1.25, j_size=1.25, i_resolution=10, j_resolution = 10)

            extrude = vtk.vtkLinearExtrusionFilter()
            extrude.SetInputData(roi_plane)
            extrude.SetScaleFactor(thickness)
            # extrude.SetVector([1,0,0])
            extrude.Update()
            roi_plane_with_thickness = pv.wrap(extrude.GetOutput())



            self.sl = self.mesh.clip_box(roi_plane_with_thickness.bounds, invert = False)

        # return self.slice
    
    def extract_slice_with_thickness(self, origin=[0, 0, 0], normal=[0, 0, 1], thickness = 0.2):
        
        roi_plane = pv.Plane(direction=normal, center=origin, 
                             i_size=1.25, j_size=1.25, i_resolution=10, j_resolution = 10)

        extrude = vtk.vtkLinearExtrusionFilter()
        extrude.SetInputData(roi_plane)
        extrude.SetScaleFactor(thickness)
        # extrude.SetVector([1,0,0])
        extrude.Update()
        roi_plane_with_thickness = pv.wrap(extrude.GetOutput())

        self.sblock = self.mesh.clip_box(roi_plane_with_thickness.bounds, invert = False)
    
    @property
    def mesh_normalization(self):
        def normalize(coord):
            return (coord - np.min(coord)) / (np.max(coord) - np.min(coord))
        return normalize

    def center_and_normalize_mesh(self):
        self.mesh.points -= self.mesh.center
        normalize = self.mesh_normalization
        self.mesh.points[:, 0] = normalize(coord=self.mesh.points[:, 0])
        self.mesh.points[:, 1] = normalize(coord=self.mesh.points[:, 1])
        self.mesh.points[:, 2] = normalize(coord=self.mesh.points[:, 2])
        return self.mesh
    
    def covert_point_data_to_cell_data(self):
        return self.mesh.point_data_to_cell_data()
        
    
    def extract_slice_vertex_and_face_data(self, scalar_to_plot = 'ux'):

        """
        Gathering vertex (point) and face (cell) data from a 2D slice
        """

        ## Gathering slice point location and cell connectivity

        vertices, faces, scalars= [], [], []

        try:
            cells = self.sl.cell
        except:
            cells = self.mesh.cell

        for k, cell in enumerate(cells):

            point = [np.round(np.hstack((c[0], c[1], c[-1])), 4) 
                    for c in cell.points]
            vertex = np.array(point)

            vertex_id = np.array(cell.point_ids)*np.ones((vertex.shape[0], len(cell.point_ids)))

            # try:
            #     disp = self.sl[scalar_to_plot][k]*np.ones((vertex.shape[0],vertex.shape[1]))
            # except KeyError:
            #     # print(f"ValueError: '{self.scalar_to_plot}' is not a valid key in slice1.")
            disp = np.zeros((vertex.shape[0],vertex.shape[1]))

            vertices.append(vertex)
            faces.append(np.int32(vertex_id))
            scalars.append(disp)
        
        max_cols = max(arr.shape[1] for arr in faces)
        padded_arrays = [np.pad(arr, ((0, 0), (0, max_cols - arr.shape[1])), constant_values=1e9) for arr in faces]

        vertices_stack = np.vstack(vertices)
        faces_stack = np.vstack(padded_arrays)
        scalars_stack = np.vstack(scalars)
                
        return vertices_stack, faces_stack, scalars_stack
    
    
