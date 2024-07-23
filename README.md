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
