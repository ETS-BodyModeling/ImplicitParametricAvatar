# Clothed_SMPLX

**Clothed_SMPLX** is a 3D human body reconstruction framework that utilizes state-of-the-art techniques for accurate human pose and shape estimation from a single image of a clothed individual. The framework supports SMPL-X, a parametric model for 3D human representation.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
Clothed_SMPLX provides a robust solution for 3D reconstruction and avatar generation from a single image. It is designed to estimate both the pose and shape of a clothed human, enabling applications in virtual try-ons, augmented reality, and human-computer interaction.

This project leverages deep learning models, particularly SMPL-X, to handle complex challenges in 3D human modeling, ensuring real-time performance and high accuracy.

## Features
- **3D Human Pose Estimation**: Accurately estimate human pose from a single image.
- **Clothing Awareness**: Specially designed to work with images of clothed humans.
- **SMPL-X Model Integration**: Uses SMPL-X, the state-of-the-art parametric model for 3D body shape and pose representation.
- **Real-time Performance**: Optimized for efficient processing and real-time applications.

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

## Acknowledgements
This project builds upon several outstanding works in the fields of 3D human modeling and computer vision. We are grateful for the contributions of the following projects and tools:

- **PIFuHD**: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization. [GitHub](https://github.com/facebookresearch/pifuhd)  
- **SMPL-X**: A unified body model that captures the pose and shape of the whole human body, including the face and hands. [Website](https://smpl-x.is.tue.mpg.de/)  
- **X-Avatar**: A tool for creating photorealistic avatars. [GitHub](https://github.com/Skype-line/X-Avatar)  
- **PeopleSnapshot**: A dataset and methodology for capturing clothed human subjects in 3D. [Website](https://peoplesnapshot.is.tue.mpg.de/)  
- **LaMa**: An open-source framework for image inpainting. [GitHub](https://github.com/saic-mdal/lama)

Their contributions have been invaluable in the development of Clothed_SMPLX. We deeply appreciate their efforts in advancing research and providing open-source tools to the community.

