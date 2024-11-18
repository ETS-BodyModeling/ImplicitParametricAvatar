# Clothed_SMPLX

## Publication

This project is associated with the research paper:

**"Implicit and Parametric Avatar Pose and Shape Estimation From a Single Frontal Image of a Clothed Human"**  
Authors:  
- **Fares Mallek** ([ORCID](http://orcid.org/0009-0001-1221-4431))  
- **Carlos Vázquez** ([ORCID](http://orcid.org/0000-0003-2161-8507))  
- **Eric Paquette** ([ORCID](http://orcid.org/0000-0001-9236-647X))  

**Clothed_SMPLX** In this work, we present a novel approach for three-dimensional estimation of human avatars, leveraging the Skinned Multi-Person Linear (SMPL-X) parametric body model. Our methodology integrates the strengths of:
1. A **Pixel-aligned Implicit Function (PIFuHD)** model for mesh generation.
2. A multi-step optimization process to infer accurate SMPL-X parameters and deformation vector for clothing.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Acknowledgements](#acknowledgements)


## Overview
Clothed_SMPLX provides a robust solution for 3D reconstruction and avatar generation from a single image. It is designed to estimate both the pose and shape of a clothed human, enabling applications in virtual try-ons, augmented reality, and human-computer interaction.

This project leverages deep learning models, particularly SMPL-X, to handle complex challenges in 3D human modeling, ensuring real-time performance and high accuracy.

## Features
- **2D Joint Estimation**: Using a pose estimation tool to recover 2D joint positions.  
- **3D Joint Estimation**: Combining 2D poses with PIFu-generated meshes to estimate 3D joint positions.  
- **Robust SMPL Initialization**: Rigid alignment of the SMPL model to the 3D joints with global translation and rotation optimization.  
- **Refinement**: Enhanced parameter refinement using new loss terms, such as point-to-surface and Chamfer distances, for improved model fidelity.  
- **Clothing Geometry Refinement**: Deformation vector fields to model tight and loose clothing geometries.  
- **Texture Mapping**: A dedicated texture extraction and completion pipeline for detailed avatar representation.

- **Real-time Animation**: Optimized for efficient processing and real-time applications.

## Installation

### Prerequisites

- Python 3.x
- PyTorch (version compatible with your system's CUDA version)
- Other dependencies are listed in the `requirements.txt`.

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/faresmallek/Clothed_SMPLX.git
   cd Clothed_SMPLX

2. ```bash
   pip install -r requirements.txt

3. ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

4. Download Preprocessed X-Avatar Data

To use the X-Avatar preprocessed data required for this project:

 Visit the Google Drive link: [X-Avatar Preprocessed Data](https://drive.google.com/drive/folders/1YRT0622s9sRmFqNLahuOP85OPLMZuG5e?usp=sharing).

Download the entire folder by clicking the **"Download"** button on the top-right of the Drive interface.

 Extract the downloaded ZIP file (if applicable) to your local directory. For instance, place the contents into a folder named `data` in the project root directory:

 5. ```bash 
    python main.py



## Acknowledgements
This project builds upon several outstanding works in the fields of 3D human modeling and computer vision. We are grateful for the contributions of the following projects and tools:

- **PIFuHD**: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization. [GitHub](https://github.com/facebookresearch/pifuhd)  
- **SMPL-X**: A unified body model that captures the pose and shape of the whole human body, including the face and hands. [Website](https://smpl-x.is.tue.mpg.de/)  
- **X-Avatar**: A tool for creating photorealistic avatars. [GitHub](https://github.com/Skype-line/X-Avatar)  
- **PeopleSnapshot**: A dataset and methodology for capturing clothed human subjects in 3D. [Website](https://peoplesnapshot.is.tue.mpg.de/)  
- **LaMa**: An open-source framework for image inpainting. [GitHub](https://github.com/saic-mdal/lama)

Their contributions have been invaluable in the development of Clothed_SMPLX. We deeply appreciate their efforts in advancing research and providing open-source tools to the community.


### Funding
This research was supported by the following funding sources:
- **MITACS** (IT19934)  
- **NSERC** (RGPIN-2021-04293 and RGPIN-2019-05252)

### Citation

If you use this project in your research, please cite:
```bibtex
@inproceedings{mallek2024implicit,
  title={Implicit and Parametric Avatar Pose and Shape Estimation From a Single Frontal Image of a Clothed Human},
  author={Mallek, Fares and V{\'a}zquez, Carlos and Paquette, Eric},
  booktitle={Proceedings of the 17th ACM SIGGRAPH Conference on Motion, Interaction, and Games},
  pages={1--11},
  year={2024}
}
