python3 eval_local.py \
    --model_arch "deeplabv3plus" \
    --model_path "./checkpoints/DeepLab_CrossEntropy_lr/best_model-epoch=0097-val_loss=0.21301333792507648.pt" \
    --Resnet_size 101 \
    --freeze_backbone

python3 eval_local.py \
    --model_arch "deeplabv3plus" \
    --model_path "./checkpoints/DeepLab_Focal_lr_64/final_model-epoch=0099-val_loss=0.09738.pt" \
    --Resnet_size 101 \
    --freeze_backbone


python3 eval_local.py \
    --model_arch "unet" \
    --model_path "./checkpoints/UNet_CrossEntropy_64/best_model-epoch=0090-val_loss=0.18080894090235233.pt" \

python3 eval_local.py \
    --model_arch "unet" \
    --model_path "./checkpoints/UNet_Focal_64/best_model-epoch=0087-val_loss=0.09473561774939299.pt" \