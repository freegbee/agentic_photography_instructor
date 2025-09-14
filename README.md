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
## Aesthetic Predictor:
- Source repository: https://github.com/discus0434/aesthetic-predictor-v2-5/tree/main

## Bounding Boxes and Object Detection
- Can be used for cropping to main object
- Possibly introduce a cache for images that already have a bounding box?
  - key could be hash value of image?
  - value could be the bounding box/bounding boxes?
- Use an existing model for bounding box detection
  - Using YOLOv8 for object detection? https://docs.ultralytics.com/de/tasks/detect/
  - Using a pre-trained model from torchvision? https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
    - Faster R-CNN
    - Mask R-CNN
    - RetinaNet
    - SSD
  

## Image Manipulation
- Use opencv for image manipulation: https://opencv.org/

### Specific Manipulation

### Cropping
#### General Information
- [Cropping an Image Using OpenCV](https://opencv.org/blog/cropping-an-image-using-opencv/#h-image-cropping-syntax-using-opencv)

#### Specific Cropping Techniques
- Cropping to aspect ratio
- Copping 1:1 in the center
- Moving window for 16x9 crops (if image is not already 16x9)
- Cropping to bounding box of main object (using bounding box detection model)

### Flipping

### Rotations

### Color Adjustments
#### General Information: 
- [Grayscaling of Images unsing OpenVP](https://www.geeksforgeeks.org/python/python-grayscaling-of-images-using-opencv/)
- [Exploring Color Manipulation in Images with OpenCV: Masking vs. Custom Color Selection](https://medium.com/@sunil.veera/exploring-color-manipulation-in-images-with-opencv-masking-vs-custom-color-selection-803d9742969c)
- [single image dehaze](https://github.com/He-Zhang/image_dehaze)
- [De-Haze](https://github.com/paulrounak/De-Haze)
- [Single Image Haze Removal Using Dark Channel Prior](https://people.csail.mit.edu/kaiming/publications/cvpr09.pdf)
- [DehazeNet: An End-to-End System for Single Image Haze Removal](https://openaccess.thecvf.com/content_cvpr_2016/papers/Cai_DehazeNet_An_End-to-End_CVPR_2016_paper.pdf)
- [AOD-Net: All-in-One Dehazing Network](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_AOD-Net_All-In-One_Dehazing_ICCV_2017_paper.pdf)

#### Specific Color Adjustments
- Grayscaling
- Color Enhancements
- Brightness Adjustments
- Contrast Adjustments
- Saturation Adjustments
- Hue Adjustments
- Color extraction
- Dehaze

## Image Database cloning


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
- [Torch nicht mit CUDA aktiviert: Ursachen und Lösungen](https://www.nnscript.de/torch-nicht-mit-cuda-aktiviert-ursachen-und-loesungen/)

## Git LFS
- Use [git lsf](https://git-lfs.com) for binarized files like images and models.
- Install git lfs: https://git-lfs.com/
- Initialize git lfs in the repository: `git lfs install`
- Track specific file types: `git lfs track "*.pth"` (for PyTorch models)
- The .gitattributes file needs to be checked in.

## Docker
- If we (need/want) use vast.ai, we also would need ro be able to use docker for deployment
  - Hence some docker related files are needed
  - And the images (input/output) need to be transferred "out of the container"

## Monitoring of Training and other long-running processes
- We could maybe use pytorch lightning for monitoring the training process.
- Or possibly "roll your own" with https://www.datacamp.com/tutorial/grafana-tutorial-monitoring-machine-learning-models?dc_referrer=https%3A%2F%2Fduckduckgo.com%2F

## Multiple pretrained models
- We could (and will) use multiple pretrained models for different tasks, like bounding box detection, aesthetic prediction etc.
- We need an efficient way to load these models, possibly with a cache.
- We need to make sure that the models are compatible with the pytorch version and CUDA version we use (if we use CUDA).
- The transformers (and the juror) will need to access the pretrained models
- See [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) from pytorch

## Python Programming
### Pickle and other Efficient File Management Tasks/Options
- We have image data and metadata to the image (e.g. aesthetic score, label) and we want it be somehow combined.
- Use pickle to serialize/deserialize objects to/from files
- [Best Practice for Data Formats in Deep Learning](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/56295498/Best+Practice+for+Data+Formats+in+Deep+Learning)
- [Is there an analysis speed or memory usage advantage to using HDF5 for large array storage (instead of flat binary files)?](https://stackoverflow.com/a/27713489)
- [Loading pickle files with pytorch dataloader](https://discuss.pytorch.org/t/loading-pickle-files-with-pytorch-dataloader/129405)
- [Python Pickle Module: A Comprehensive Guide](https://realpython.com/python-pickle-module/)
- [Welcome to compress_pickle’s documentation!](https://lucianopaz.github.io/compress_pickle/html/)
- [h5py](https://docs.h5py.org/en/stable/quick.html#quick) um HDF5 Dateien (grosse numpy Datenmengen, z.B. unsere Bilde) zu lesen und schreiben
- [lmdb](https://lmdb.readthedocs.io/en/release/) alternativer Ansatz, um grosse Datenmengen effizient zu speichern und zu laden
- [Zarr](https://zarr.readthedocs.io/en/stable/) another alternative for large data storage


### Singleton Pattern
- [Singleton Pattern in Python](https://refactoring.guru/design-patterns/singleton/python/example) - Check out "threadsafe singleton"
  - Could be used for model loading and caching, threadsafe printing/logging etc.
- [Singleton Pattern in Python - A Complete Guide](https://www.geeksforgeeks.org/python/singleton-pattern-in-python-a-complete-guide/)

### Class Registry
- We are currently using [class-registry](https://github.com/todofixthis/class-registry)

### Decorators
- [Annotations Best Practices](https://docs.python.org/3/howto/annotations.html)
- [Primer on Python Decorators](https://realpython.com/primer-on-python-decorators/#registering-plugins)
- [Going Beyond Decorators: Creating a Custom Python Annotation System](https://jsschools.com/python/going-beyond-decorators-creating-a-custom-python-/)
- Use for timing and logging around function calls (for prometheus, grafana, lightning-torch logging of training progress etc.)
- Use with threadsafe printer/singleton to loge before and after function calls

## Thinking further
### Training Image generation
- Our current approch is, to define the transformation sequence up front and then select the best image. This could also be changed to a dynamic approach, where we train a nural network to generate the tranformations tha promise the highes score increase.
- Suggestions according to copilot:
  - We could use a "transformer graph" instead of a linear sequence of transformers. This would allow for more complex transformations and possibly better results.
  - We could use reinforcement learning to train the agents, where the reward is based on the aesthetic score of the final image.
  - We could use a generative model (like a GAN) to generate new images based on the transformed images.
  - We could use a multi-agent system where multiple agents work together to transform the image.
  - We could use a "jury" of models to evaluate the transformed images and provide feedback to the agents.
  - We could use a "curriculum learning" approach where the agents start with simple transformations and gradually learn more complex ones.
  - We could use a "self-play" approach where agents compete against each other to produce the best images.

### Agents
- [A2A Protocol](https://a2a-protocol.com)