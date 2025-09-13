# Agentic Photography Instructor

## Introduction
Agentic Photographer Instructor is a project idea for the CAS "Machine Learning for Software Engineers" 2025/2026 at the OST in Rapperswil, Switzerland.


## Project Structure
- `src`: Source code for the project
- `test`: testing resources and utilities
- `configs`: Configuration files
- `data`: Datasets (samples, not main dataset as it is not part of the python project)
- `resources`: Additional resources (e.g., test images, documentation)
- `temp_output`: Temporary output files (e.g., generated images)
- `notebooks`: Jupyter notebooks for experiments and analysis
- `env`: Local, not-checked-in files (e.g. local overrides of configuration properties)

### Source Code Structure (src)
- types: Shared data types
- utils: utility functions
- transformer: All transformer functionality, e.g. cropping, color enhancing etc.

## Implementation
### Transformers
- Transformers are changing the opencv image (opencv image is a numpy array).
- Each transformer must inherit from `AbstractTransformer` and implement the transform method.
- Each transformer must declare the static properties `label` (str), `description` (str) and `transformer_type`.
  - `label` is a short, unique string. The labels are (part) the labels the NN will learn
  - `description` is a longer description of the transformer.
  - `transformer_type` is an enum of type TransformerTypeEnum which categorizes the transformer
- Transformers can be derived from an abstract class "in the middle" that defines the `transformer_type`. In this case, the abstract class in the middle must be abstract as well. To ensure this, it needs to deklase a method as @abstractmethod (sic!)
- Transformers will be autoregistered. To facilitate this, they need to be imported in `__init__.py` of their respective transformer package and exported.
- All modules with concrete transformers need to be imported in `src/transformer/__init__.py` and exported.

### TransformationAgent and AgentFactories
- A `TransformationAgent` is a class that applies a series of transformers to an image. Agent has a fixed sequence of transformers. The image is sequentially passed from one transformer to the next. Provided image is _not_ copied, thus before calling the agent the image data must be copied/cloned.
- The `TransformationAgent` recieves a list of transformer keys in its constructor. Thus it can be created with a fixed list of transformers to create training data. Or it can be based on a predicted label coming from the trained NN. 
- A `AbstractTransformationAgentFactory` is a factory that creates a list of `TransformationAgent` instances. THis results in a list of agents that each has a list of transformers.
- A concrete factory must inherit from `AbstractTransformationAgentFactory` and implement the `create_agents` method.
- The concrete factory is auto registering. To facilitate this, it needs to be imported in `src/transformer_agent/__init__.py`
- Concrete factories allow for flexible composition, i.e. create a factory with a static list of agents or a factory that reads a CSV with all the agents to be created with different transformations 
- An example of a concrete factory is `StaticTransformationAgentFactory`

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