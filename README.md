# Implicit and Parametric Avatar Pose and Shape Estimation From a Single Frontal Image of a Clothed Human

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DF2qVcJJMTN7scutey1qfUF_LeYsf7EW?usp=sharing) [![ACM Publication](https://img.shields.io/badge/ACM-MIG%252024-blue)](https://dl.acm.org/doi/10.1145/3677388.3696328) 
[![View in Browser](https://img.shields.io/badge/View-HTML%20Version-orange)](https://camps.aptaracorp.com/ACM_PMS/PMS/ACM/MIG24/9/a6c2ee09-7446-11ef-ada9-16bb50361d1f/OUT/mig24-9.html)

## Publication

This project is associated with the research paper:
Fares Mallek ([ORCID](http://orcid.org/0009-0001-1221-4431)), Carlos Vázquez ([ORCID](http://orcid.org/0000-0003-2161-8507)), and Eric Paquette ([ORCID](http://orcid.org/0000-0001-9236-647X)). 2024. **Implicit and Parametric Avatar Pose and Shape Estimation From a Single Frontal Image of a Clothed Human**. In Proceedings of the 17th ACM SIGGRAPH Conference on Motion, Interaction, and Games (MIG '24). Association for Computing Machinery, New York, NY, USA, Article 20, 1–11. https://doi.org/10.1145/3677388.3696328

![Teaser Image](https://github.com/ETS-BodyModeling/ImplicitParametricAvatar/blob/main/teaser/intro_fig.png)
![Demo GIF](https://github.com/ETS-BodyModeling/ImplicitParametricAvatar/blob/main/teaser/rotation.gif)
![Demo GIF](https://github.com/ETS-BodyModeling/ImplicitParametricAvatar/blob/main/teaser/animation_1.gif)


In this work, we present a novel approach for three-dimensional estimation of human avatars, leveraging the Skinned Multi-Person Linear (SMPL-X) parametric body model. Our methodology integrates the strengths of:
1. A Pixel-aligned Implicit Function (PIFuHD) model for mesh generation.
2. A multi-step optimization process to infer accurate SMPL-X parameters and deformation vector for clothing.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick testing](#quick_testing)
- [Testing](#testing)
- [Acknowledgements](#acknowledgements)


## Overview
This project provides a robust solution for 3D reconstruction and avatar generation from a single image. It is designed to estimate both the pose and shape of a clothed human, enabling applications in virtual try-ons, augmented reality, and human-computer interaction.

## Features
- **2D Joint Estimation**: Using a pose estimation tool to recover 2D joint positions.  
- **3D Joint Estimation**: Combining 2D poses with PIFu-generated meshes to estimate 3D joint positions.  
- **SMPL-X Shape and Pose Optimization**: A multi-step optimization aligns the SMPL-X body to the PIFuHD mesh. 
- **Clothing Geometry Refinement**: Per vertex deformation vectors to model tight and loose clothing geometries.  
- **Texture Mapping**: A dedicated texture extraction and completion pipeline for detailed avatar representation.

Here is an example of an animation:
![Demo GIF](https://github.com/ETS-BodyModeling/ImplicitParametricAvatar/blob/main/teaser/animation.gif)

## Installation

**For Linux**:

### Prerequisites

- Python 3.10
- PyTorch (version compatible with your system's CUDA version)
- Pytorch3d (version compatible with your system's CUDA version)
- Other dependencies are listed in `requirements.txt`

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/ETS-BodyModeling/ImplicitParametricAvatar.git
   cd ImplicitParametricAvatar
   ```
2. ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   ```
3. ```bash
   pip install -r requirements.txt
   ```

4. Setup submodules , run this command:
```bash
   git submodule update --init --recursive
   ```


5. Download Pifuhd model
   ```bash
   sh ./submodules/pifuhd/scripts/download_trained_model.sh
   ```

6. Download the required .npz files (for both male SMPLX_MALE.npz and female models SMPLX_FEMALE.npz) from the official [SMPL-X website](https://smpl-x.is.tue.mpg.de/), place the contents into a folder named `models/smplx` in the project root directory

## Quick_testing
To quickly run the demo, use the following command:
   ```bash
   python -m scripts.demo
   ```
The pipeline loads a sample image and its corresponding pose from OpenPose (in /data/demo), runs pifuhd, the reconstruction process and generate animation. Displays the generated output result (in the output/ directory).

## testing
1. Download Preprocessed X-Avatar Data

To use the X-Avatar preprocessed data required for this project:

 Visit the Google Drive link: [X-Avatar Preprocessed Data](https://drive.google.com/drive/folders/1YRT0622s9sRmFqNLahuOP85OPLMZuG5e?usp=sharing).

Download the entire folder by clicking the **"Download"** button on the top-right of the Drive interface.

 Extract the downloaded ZIP file (if applicable) to your local directory. For instance, place the contents into a folder named `data` in the project root directory:


2. To run the reconstruction and texture extraction pipeline, execute the following command

   ```bash
   python -m scripts.main_optim  --input_path data/recon
   ```

3. To render a frontal view of the reconstructed 3D model, use the following script:

   ```bash
   python -m scripts.main_render
   ```

4. To generate statistics about the input images and reconstructed meshes, download meshes obj target from X-Avatar website and use the following command:
   ```bash  
   python -m scripts.main_stats 
   ```
5. To Generate Animations from Reconstructed Meshes

   ```bash  
   python -m scripts.main_animation 
   ```
This script applies motion sequences from the AMASS dataset to the reconstructed meshes, creating dynamic animations.

## Acknowledgements
This project builds upon several works in the fields of 3D human modeling and computer vision. We are grateful for the contributions of the following projects and tools:

- **PIFuHD**: Pixel-Aligned Implicit Function for High-Resolution  Human Digitization. [GitHub](https://github.com/facebookresearch/pifuhd)  
- **SMPL-X**: A unified body model that captures the pose and shape of the whole human body, including the face and hands. [Website](https://smpl-x.is.tue.mpg.de/)  
- **X-Avatar**: A tool for creating photorealistic avatars. [GitHub](https://github.com/Skype-line/X-Avatar)  
- **PeopleSnapshot**: A dataset and methodology for capturing  human subjects in 3D. [Website](https://graphics.tu-bs.de/people-snapshot)  
- **LaMa**: An open-source framework for image inpainting. [GitHub](https://github.com/saic-mdal/lama)
- **AMASS**: Archive of Motion Capture as Surface Shapes, a dataset for high-quality motion sequences and animations [Website](https://paperswithcode.com/dataset/amass) 

### Funding
This research was supported by the following funding sources:
- **MITACS** (IT19934)  
- **NSERC** (RGPIN-2021-04293 and RGPIN-2019-05252)

## License

This work is licensed under a [Creative Commons Attribution International 4.0 License](https://creativecommons.org/licenses/by/4.0/).

MIG '24, November 21–23, 2024, Arlington, VA, USA

© 2024 Copyright held by the owner/author(s).  
ACM ISBN 979-8-4007-1090-2/24/11.  
DOI: [https://doi.org/10.1145/3677388.3696328](https://doi.org/10.1145/3677388.3696328)


### Citation

If you use this project in your research, you must cite:
```bibtex
@inproceedings{10.1145/3677388.3696328,
  author = {Mallek, Fares and V\'{a}zquez, Carlos and Paquette, Eric},
  title = {Implicit and Parametric Avatar Pose and Shape Estimation From a Single Frontal Image of a Clothed Human},
  year = {2024},
  isbn = {9798400710902},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3677388.3696328},
  doi = {10.1145/3677388.3696328},
  booktitle = {Proceedings of the 17th ACM SIGGRAPH Conference on Motion, Interaction, and Games},
  articleno = {20},
  numpages = {11},
  keywords = {3D modeling, Human avatars, animation, computer vision., deep neural networks, parametric models, textures},
  location = {Arlington, VA, USA},
  series = {MIG '24}
}

