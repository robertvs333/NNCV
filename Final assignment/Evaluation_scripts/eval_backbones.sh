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
    --model_arch "deeplabv3plus" \
    --model_path "./checkpoints/DeepLab_CrossEntropy_64/best_model-epoch=0096-val_loss=0.25622949562966824.pt" \
    --Resnet_size 101 \

python3 eval_local.py \
    --model_arch "deeplabv3plus" \
    --model_path "./checkpoints/DeepLab_Focal_64/best_model-epoch=0098-val_loss=0.11932653281837702.pt" \
    --Resnet_size 101 \

python3 eval_local.py \
    --model_arch "deeplabv3plus" \
    --model_path "./checkpoints/DeepLab_50_CrossEntropy_lr/best_model-epoch=0098-val_loss=0.2236898336559534.pt" \
    --Resnet_size 50 \
    --freeze_backbone

python3 eval_local.py \
    --model_arch "deeplabv3plus" \
    --model_path "./checkpoints/DeepLab_50_Focal_lr/best_model-epoch=0093-val_loss=0.11072919797152281.pt" \
    --Resnet_size 50 \
    --freeze_backbone


python3 eval_local.py \
    --model_arch "deeplabv3plus" \
    --model_path "./checkpoints/DeepLab_50_CrossEntropy_64/best_model-epoch=0098-val_loss=0.2694107014685869.pt" \
    --Resnet_size 50 \

python3 eval_local.py \
    --model_arch "deeplabv3plus" \
    --model_path "./checkpoints/DeepLab_50_Focal_64/best_model-epoch=0093-val_loss=0.13070760667324066.pt" \
    --Resnet_size 50 \