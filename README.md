# Agentic Photography Instructor

## Introduction
Agentic Photographer Instructor is a project idea for the CAS "Machine Learning for Software Engineers" 2025/2026 at the OST in Rapperswil, Switzerland.


## Project Structure
- src: Source code for the project
- test: testing resources and utilities
- configs: Configuration files
- data: Datasets (samples, not main dataset as it is not part of the python project)
- resources: Additional resources (e.g., test images, documentation)
- notebooks: Jupyter notebooks for experiments and analysis
- end: Local, not-checked-in files (e.g. local overrides of configuration properties)

### Source Code Structure (src)
- types: Shared data types
- utils: utility functions
- transformer: All transformer functionality, e.g. cropping, color enhancing etc.


# Machine Learning Related Core Information
- Aesthetic Predictor:
  - Source repository: https://github.com/discus0434/aesthetic-predictor-v2-5/tree/main

# Additional Notes
## CUDA Support
- For NVIDIA CUDA Support, install the [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx) and [CUDA Toolkit](https://developer.nvidia.com/cuda-12-9-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local).
  - As currently, pytorch only supports CUDA 12.9, we're installing that version 
- Proper PyTorch version installation with CUDA support: https://pytorch.org/get-started/locally/
  - Make sure CUDA is correct version for pytorch.
- Checking of prerequisites:
  - `nvidia-smi` in cmd should show the installed driver version
  - `nvcc --version` in cmd should show the installed CUDA version
  - In python, run:
    ```python
    import torch
    print(torch.cuda.is_available()) # should return True
    print(torch.cuda.current_device()) # should return 0
    print(torch.cuda.get_device_name(0)) # should return the name of your GPU
    ```
## Git LFS
- Use [git lsf](https://git-lfs.com) for binarized files like images and models.
- Install git lfs: https://git-lfs.com/
- Initialize git lfs in the repository: `git lfs install`
- Track specific file types: `git lfs track "*.pth"` (for PyTorch models)
- The .gitattributes file needs to be checked in.