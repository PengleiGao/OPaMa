import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import shutil

outdoor_ckpt='Outpaint_log_mamba/mamba_outdoor/lighting_logs/version_yk/latest_checkpoint.ckpt'

cmd=f'python inference.py -b configs/eval_outdoor_mamba.yaml --checkpoint {outdoor_ckpt} --inference_out_dir Results/version_2/outdoor --num_inference_step 25 --cfg_scale 2.5 --name my_project'
os.system(cmd)
