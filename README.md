# Parametric Model Fitting for Textured and Animatable 3D Avatar From a Single Frontal Image of a Clothed Human

**Under review for the Graphics Replicability Stamp Initiative (GRSI)**

---

## GRAPHICS REPLICABILITY STAMP INITIATIVE (GRSI)

This repository is provided for evaluation by the **Graphics Replicability Stamp Initiative** (GRSI).  
All materials required for the review are located in the root directory or referenced below.

### Repository Contents

- **GRSI Submission Information:** [`submission_info.txt`](submission_info.txt)  
  Contains the submission title, authors, and operating system, as required for GRSI.
- **GRSI Liability Form:** [`LIABILITY_FORM.md`](LIABILITY_FORM.md)  
  Provides permission for the reproducibility committee and reviewers to evaluate and publicly advertise the review.
- **GRSI Installation Script:** [`INSTALL.sh`](INSTALL.sh)  
  Automates environment setup and dependency installation on a vanilla system. See [Installation & Setup](#installation--setup) for details.
- **GRSI Result Generation Scripts:**  
  Scripts to reproduce the quantitative results presented in the paper (e.g., *Table 1*). These scripts run without parameters. 
 Specifically, running python -m scripts.run_pipeline_simple and python -m scripts.run_pipeline_simple_sub will generate the final results in a file named output_simple/ours.csv and output_sub/ours.csv.

---

## Publication

This repository accompanies the following publication:

**Fares Mallek**, **Carlos Vázquez**, and **Eric Paquette**.  
*Parametric Model Fitting for Textured and Animatable 3D Avatar From a Single Frontal Image of a Clothed Human.*  
Accepted in *Computers & Graphics*, 2024.

**DOI:** _Under review_  
**Associated Work:**  
> Mallek, F., Vázquez, C., & Paquette, E. (2024). *Implicit and Parametric Avatar Pose and Shape Estimation From a Single Frontal Image of a Clothed Human.*  
> In *Proceedings of Motion, Interaction and Games (MIG’24)*, November 21–23, 2024, Arlington, VA, USA.  
> [https://doi.org/10.1145/3677388.3696328](https://doi.org/10.1145/3677388.3696328)

---

## Overview

In this work, we introduces an easily animatable new SMPL-X mesh topology with shoe-like feet, enabling more realistic and flexible character animation. Key features include:

- **New SMPL-X Mesh Topology**: Designed for easy animation, the mesh features shoe-like feet for improved realism and articulation.
![Improvement1](https://github.com/ETS-BodyModeling/ImplicitParametricAvatar/blob/main/teaser/feet_improvement.png)
- **Subdivision for Detail**: The SMPL-X mesh is subdivided to enhance the representation of fine details, especially when computing deformation vectors.

These improvements make the model more suitable for high-quality animation and rendering tasks.
![Improvement2](https://github.com/ETS-BodyModeling/ImplicitParametricAvatar/blob/main/teaser/improvement.png) 

---

## Features

- **2D Joint Estimation:** Recovery of body keypoints using a 2D pose estimator.  
- **3D Joint Estimation:** Combination of 2D poses with PIFu-generated meshes for 3D joint reconstruction.  
- **SMPL-X Shape & Pose Optimization:** Multi-stage optimization aligning the SMPL-X model with PIFuHD reconstructions.  
- **Clothing Geometry Refinement:** Per-vertex deformation modeling of tight and loose clothing.  
- **Texture Mapping:** Dedicated extraction and completion pipeline for detailed, animatable texture maps.  
- **Enhanced SMPL-X Topology:** A subdivided, shoe-like mesh topology for improved realism and articulation.

---

## GRSI Replicability

The repository is designed for reproducibility under the **Graphics Replicability Stamp Initiative (GRSI)** standards.  
All quantitative results (e.g., Table 1) can be regenerated through the provided scripts.  

---

## Installation & Setup
### Prerequisites
- Python 3.10
- PyTorch (version compatible with your system's CUDA version)
- Pytorch3d (version compatible with your system's CUDA version)
- Other dependencies are listed in `requirements.txt`
### Hardware Prerequisites
- NVIDIA GPU (GeForce RTX 3060 or better recommended)

### Tested Environments

**System 1**
- OS: Ubuntu 24.04.2 LTS  
- GPU: NVIDIA GeForce RTX 3060  
- Driver: 550.120  
- CUDA: 12.4  

**System 2**
- OS: Amazon Linux 2023.6.20241111  
- GPU: NVIDIA L4  
- Kernel: 6.1.115-126.197.amzn2023.x86_64  
- CUDA: 12.6  

---

### Setup Instructions

Setup involves three main steps:

#### 1. Automated Dependency Installation (GRSI)

Run the installation script:

```bash
bash INSTALL.sh
source .venv/bin/activate
```
2. Manual Download — SMPL-X Models

Download the SMPL-X models from the official website
.
Place the .npz files in:

models/smplx/

  ├── SMPLX_MALE.npz

  └── SMPLX_FEMALE.npz


1. Download Preprocessed X-Avatar Data

To use the X-Avatar preprocessed data required for this project:

 Visit the Google Drive link: [X-Avatar Preprocessed Data](https://drive.google.com/drive/folders/1YRT0622s9sRmFqNLahuOP85OPLMZuG5e?usp=sharing).

Download the entire folder by clicking the **"Download"** button on the top-right of the Drive interface.

 Extract the downloaded ZIP file (if applicable) to your local directory. For instance, place the contents into a folder named `data` in the project root directory:

2. To run the simple SMPL-X reconstruction pipeline, perform texture extraction, rendering, and evaluate the shoe-like method, execute the following command (it takes around **1 hour on NVIDIA L4** or **30 minutes on RTX 3060** for 20 X-Avatar input data without the animation flag).
   
  GRSI: This code reproduces the "Ours" entry from Table 1 of our paper:

   ```bash
   python -m scripts.run_pipeline_simple
   ```
To create animation from the reconstructed data, add the `--run_animation` flag:
   ```bash
   python -m scripts.run_pipeline_simple --run_animation
   ```

   The results will be saved in the `output_simple` directory with the following subfolders:

   - `recons`: Contains reconstructed SMPL-X meshes and corresponding textures.
   - `output_render`: Stores rendered frontal images, normal maps, and silhouettes.
   - `stats`: Includes quantitative statistics evaluating the performance of the method.
   - `output_animation`: The resulting animation will be saved in this directory, showcasing the avatar animated with the AMASS hiphop sequence.


3. To run the shoe-like SMPL-X subdivided reconstruction pipeline, perform texture extraction, rendering, and animation, execute the following command (it takes around **2 hour on NVIDIA L4** or **1 hour on RTX 3060** for 20 X-Avatar input data without the animation flag).
   
   GRSI: This code reproduces the "Ours subdiv " entry from Table 1 of our paper:

   ```bash
   python -m scripts.run_pipeline_simple_sub
   ```
To create animation from the reconstructed data, add the `--run_animation` flag:
   ```bash
   python -m scripts.run_pipeline_simple_sub --run_animation
   ```
   The results will be saved in the `output_sub` directory with the following subfolders:

   - `recons`: Contains reconstructed SMPL-X meshes and corresponding textures.
   - `output_render`: Stores rendered frontal images, normal maps, and silhouettes.
   - `stats`: Includes quantitative statistics evaluating the performance of the method.
   - `output_animation`: The resulting animation will be saved in this directory, showcasing the avatar animated with the AMASS hiphop sequence.

This script applies motion sequences from the AMASS dataset to the reconstructed meshes, creating dynamic animations. -->

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
