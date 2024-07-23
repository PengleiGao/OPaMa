import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from cleanfid.clip_features import CLIP_fx, img_preprocess_clip

clip_fx = CLIP_fx("ViT-B/32")