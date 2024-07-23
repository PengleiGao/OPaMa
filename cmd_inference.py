import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import shutil

# baseline FID 18.4 IS 4.2 LPIPS 0.36
#
# ckpt='/mnt/e/Outpaint/mamba_outdoor/lightning_logs/version_4/checkpoints/latest_checkpoint.ckpt'
# save_dir='/mnt/e/Datasets/outdoor_wo_text'
# cmd=f'python inference_wo_text.py -b configs/eval_outdoor_mamba.yaml --checkpoint {ckpt} --inference_out_dir {save_dir} --num_inference_step 50 --cfg_scale 2.5'
#os.system(cmd)


outdoor_ckpt='Outpaint_log_mamba/mamba_outdoor/lighting_logs/version_yk/latest_checkpoint.ckpt'
#outdoor_ckpt='Outpaint_log_mamba/mamba_outdoor/lightning_logs/version_2/checkpoints/epoch=239-train/loss=0.04.ckpt'


# cmd=f'python inference.py -b configs/eval_outdoor_mamba.yaml --checkpoint {outdoor_ckpt} --inference_out_dir Results/version_2/outdoor_DDPM_both --num_inference_step 25 --cfg_scale 2.5 --name my_project_239'
# os.system(cmd)

# cmd=f'python inference_wo_text.py -b configs/eval_outdoor_mamba.yaml --checkpoint {outdoor_ckpt} --inference_out_dir /mnt/e/Datasets/outdoor_with_image --num_inference_step 50 --cfg_scale 2.5'
# os.system(cmd)

cmd=f'python inference_with_only_text_outdoor.py -b configs/eval_outdoor_mamba.yaml --checkpoint {outdoor_ckpt} --inference_out_dir Results/version_yk/outdoor_with_text --num_inference_step 25 --cfg_scale 2.5 --name my_project_light2'
os.system(cmd)