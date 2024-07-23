# Opa-Ma-360-degree-Image-Outpainting-with-Mamba
official implementation for the paper "OPa-Ma: Text Guided Mamba for 360-degree Image Out-painting"

## Highlights
• We design a Visual-textual Consistency Refiner (VCR) to produce a better con-
ditional context with the input of text guidance and image guidance. The con-
ditional context is obtained with a weighted sum of the modified image feature
and text feature with stacked 1D Mamba blocks, providing consistency refining.

• We develop a Global-local Mamba Adapter (GMA) to extract the global and
local features and connect information flow among the NFoV images with the
characteristic of selective state space of Mamba. The local feature is captured
of the input image based on the global state representations that are extracted
among the multi-direction NFoV images.

• Extensive experiments achieve state-of-the-art performance on two widely rec-
ognized 360-degree image datasets in indoor and outdoor settings indicating the
the superiority of our method.

![We aim to generate smooth and reasonable 360-degree images from NFoV images by utilizing a state space model Mamba with the ability to process long sequences and model spatial continuity.](/assets/prebanner.png)

We aim to generate smooth and reasonable 360-degree images from NFoV images by utilizing a state space model Mamba with the ability to process long sequences and model spatial continuity.

## Requirements
you can install the required environment by using `pip install -r requirement.txt`
python>=3.8, as well as pytorch>=2.0, torchvision>=0.8, diffusers>=0.21.0 and accelerate>=0.19.0.

## Dataset
We use the outdoor and indoor datasets in our paper. The datasets can be inquired in [Laval Indoor HDR](http://hdrdb.com/indoor/) and [Laval Outdoor HDR](http://hdrdb.com/outdoor/).

## Experiments
### Preprocessing - exr
The images in these two datasets are in .exr format, to convert .exr images to .png, you can run

`python misc/captioning --image_dir [path_to_png_imgaes] --out_dir [path_to_text_prompt] --name [dataset_name]`

The augmented text prompt can be extracted via:

`python misc/captioning --image_dir [path_to_png_imgaes] --out_dir [path_to_text_prompt] --name [dataset_name] --stride 30 --augment `

### Training
You can run the following command to train your own model:

`python cmd_train.py`

### Inference
You can inference the trained model with:

`python cmd_inference.py`

## Citation
If you found our project is helpful, please cite us:
@article{gao2024opa,
  title={OPa-Ma: Text Guided Mamba for 360-degree Image Out-painting},
  author={Gao, Penglei and Yao, Kai and Ye, Tiandi and Wang, Steven and Yao, Yuan and Wang, Xiaofeng},
  journal={arXiv preprint arXiv:2407.10923},
  year={2024}
}

## Acknowledgement
This project is built based on [[AOG-NET-360](https://github.com/zhuqiangLu/AOG-NET-360), [Diffusers](https://github.com/huggingface/diffusers), and [PyEquilib](https://github.com/haruishi43/equilib)
