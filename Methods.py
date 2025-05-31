#################### 
# @Author: Tanmay.Mukherjee  
# @Date: 2025-01-04 02:13:11  
# @Last Modified by: Tanmay.Mukherjee  
# @Last Modified time: 2025-01-04 02:13:11 
####################

import numpy as np
import os

def calculate_affine_transformation_matrix(original_plane_normal, target_plane_normal, translation_vector):
  """
  Calculates the affine transformation matrix to transform a plane with the given normal 
  vector to the target plane, including translation.

  Args:
    original_plane_normal: A 3D vector representing the normal of the original plane.
    target_plane_normal: A 3D vector representing the normal of the target plane.
    translation_vector: A 3D vector representing the translation.

  Returns:
    A 4x4 affine transformation matrix.
  """

  rotation_matrix = rotation_matrix_between_planes(original_plane_normal, target_plane_normal)

  # Create the affine transformation matrix
  affine_matrix = np.eye(4)
  affine_matrix[:3, :3] = rotation_matrix
  affine_matrix[:3, 3] = translation_vector

  return affine_matrix



def rotation_matrix_between_planes(original_plane_normal, target_plane_normal):
    """
    Calculates the rotation matrix to transform a plane with the given normal 
    vector to the target plane.

    Args:
    original_plane_normal: A 3D vector representing the normal of the original plane.
    target_plane_normal: A 3D vector representing the normal of the target plane.

    Returns:
    A 3x3 rotation matrix.
    """

    # Normalize the plane normals
    original_plane_normal = original_plane_normal / np.linalg.norm(original_plane_normal)
    target_plane_normal = target_plane_normal / np.linalg.norm(target_plane_normal)

    # Calculate the rotation axis
    rotation_axis = np.cross(original_plane_normal, target_plane_normal)

    # print(rotation_axis)

    # Handle the case where the planes are parallel 
    if np.allclose(np.linalg.norm(rotation_axis), 0):
        # If the normals are parallel and point in the same direction, no rotation is needed
        if np.allclose(original_plane_normal, target_plane_normal):
            return np.eye(3) 
        # If the normals are parallel and point in opposite directions, 
        # rotate 180 degrees around an arbitrary axis perpendicular to the plane
        else:
            # Choose an arbitrary axis perpendicular to the plane
            arbitrary_axis = np.array([1, 0, 0]) 
            
            if np.allclose(np.dot(arbitrary_axis, original_plane_normal), 0):
                arbitrary_axis = np.array([0, 1, 0]) 
            
            return rotation_matrix(arbitrary_axis, np.pi) 
    else:
        # Calculate the rotation angle
        cos_theta = np.dot(original_plane_normal, target_plane_normal)
        theta = np.arccos(cos_theta)

        return rotation_matrix(rotation_axis, theta)


def rotation_matrix(axis, angle):
    """
    Creates a rotation matrix given an axis and angle.

    Args:
    axis: A 3D vector representing the rotation axis.
    angle: The rotation angle in radians.

    Returns:
    A 3x3 rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle)
    b = 1 - np.cos(angle)
    c = np.sin(angle)

    x, y, z = axis

    return np.array([[
        a + b*x*x, b*x*y - c*z, b*x*z + c*y
    ], [
        b*x*y + c*z, a + b*y*y, b*y*z - c*x
    ], [
        b*x*z - c*y, b*y*z + c*x, a + b*z*z
    ]])

