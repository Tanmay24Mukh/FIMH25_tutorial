# In silico Phantom Simulations for AI-powered 4D Cardiac Motion Estimation
## FIMH 2025 13<sup>th</sup> Functional Imaging and Modeling of the Heart International Conference
UT Southwestern Medical Center, Dallas, TX 


This tutorial provides an introduction on generating in silico phantoms from subject-specific finite element (FE) simulations of the heart. The objective is to devise a methodology to benchmark ``ground-truth'' Cartesian displacements and thus standardize the fidelity of image-based motion mapping technologies. The repository contains the codes necessary to generate grayscale images from vtk files delineating motion of (i) a cylinder under pure torsion and (ii) mouse-specific FE simulations. As such, the attendees will be able to generate their own phantoms to design, test, and validate their own image processing methods to map four-dimensional cardiac motion.

The tutorial is outlined as follows:
- Motivation & background
- Cardiac motion benchmarking
    - Introduce mouse-specific FE simulation dataset
    - Use displacement vectors for full cardiac cycle motion representation
- Synthetic Image Generation
    - Create synthetic cardiac images across the cycle
    - Visualize short- and long-axis heart slices
- Problem 1: Synthetic Phantom
    - Simulate standard imaging via mesh-to-image rasterization (MRI)
    - Use Field II algorithm for synthetic ultrasound generation
- Problem 2: Image Registration
    - Track cardiac motion using deformable image registration
    - Demonstrate torsion estimation with a cylinder model

## Codes and Datasets
- Hands-on tutorial to visualize phantoms [`FIMH25.py`](FIMH25.py)
- Automated phantom generation code [`Create_Pahntoms.py`](Create_Phantoms.py)
- Cylinder and heart vtks [`Datasets`](Datasets)

## Related work
Our latest efforts on standardizing cardiac motion algorithms is described in [`In silico heart phantom`](https://doi.org/10.1016/j.compbiomed.2024.109065)

## Organized by
Reza Avazmohammadi <sup> 1 </sup>, Kyle Myers <sup> 1 </sup>, Tanmay Mukherjee <sup> 1 </sup> <br>
<sup> 1 </sup> Texas A&amp;M University 

### Contact
Reach out to tanmaymu@tamu.edu or rezaavaz@tamu.edu for questions, collaborations, or additional feedback. 