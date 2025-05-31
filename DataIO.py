#################### 
# @Author: Tanmay.Mukherjee  
# @Date: 2025-01-03 22:10:06  
# @Last Modified by: Tanmay.Mukherjee  
# @Last Modified time: 2025-01-03 22:10:06 
####################

import os, json
import numpy as np
import pyvista as pv
import nibabel as nib
from PIL import Image


class Write_Data():

    def write_images(img, path = os.getcwd(), fid = '.nii.gz'):
        
        """
        Write image inputs.
        """
        if not os.path.isdir(path):
            os.makedirs(path)

        file_id = os.path.join(path, fid)

        img = nib.Nifti1Image(img, affine=np.eye(4)) 

        nib.save(img, file_id) 

    def write_png_images(img, path = os.getcwd(), fid = '.png'):
        
        """
        Write png images.
        """        

        file_id = os.path.join(path, fid)

        img = Image.fromarray(img).convert("L")

        img.save(file_id)


    def write_img_specs(specs = [{}], path = os.getcwd(), fid = '.json'):

        if not os.path.isdir(path):
            os.makedirs(path)

        file_id = os.path.join(path, fid)

        with open(file_id, 'w') as f:
            json.dump(specs, f, indent=4)         

    
class Read_Data():

    def read_vtk_files(path = os.getcwd()):

        if not os.path.isdir(path):
            os.makedirs(path)

        vtk_files = []
        for root, dirs , files in os.walk(path):
            for i, file in enumerate(files):
                if file.endswith('.vtk'):
                    vtk_files.append([i, os.path.join(root, file)])

        return vtk_files

    def read_ply_files(path = os.getcwd()):

        if not os.path.isdir(path):
            os.makedirs(path)

        vtk_files = []
        for root, dirs , files in os.walk(path):
            for i, file in enumerate(files):
                if file.endswith('.ply'):
                    vtk_files.append([i, os.path.join(root, file)])

        return vtk_files


    def read_img_files(path = os.getcwd()):

        if not os.path.isdir(path):
            os.makedirs(path)

        img_files = []
        for root, dirs , files in os.walk(path):
            for i, file in enumerate(files):
                if file.endswith('.nii.gz'):
                    img_files.append([i, os.path.join(root, file)])

        return img_files
    
    def read_specs_files(path = os.getcwd()):

        if not os.path.isdir(path):
            os.makedirs(path)

        specs_files = []
        for root, dirs , files in os.walk(path):
            for i, file in enumerate(files):
                if file.endswith('.json'):
                    specs_files.append([i, os.path.join(root, file)])

        return specs_files


    def read_mesh(fid = 'mesh.vtk'):

        return pv.read(fid)
    
    def read_images(fid = 'img.nib'):
        
        nii_img  = nib.load(fid)
        nii_data = nii_img.get_fdata()

        return nii_data
    
    def read_specs(fid = 'specs.json'):
        
        with open(fid, 'r') as f:
            specs = json.load(f)
        
        return specs

    def read_files(self, path = os.getcwd(), ext = '.json'):

        if not os.path.isdir(path):
            os.makedirs(path)

        spec_files = []
        for root, dirs , files in os.walk(path):
            # print(files)
            for i, file in enumerate(files):
                if file.endswith(ext):
                    spec_files.append([i, os.path.join(root, file)])

        return spec_files
        