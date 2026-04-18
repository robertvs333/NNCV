cd ..
python3 eval_local.py \
    --model_arch "deeplabv3plus" \
    --model_path "./checkpoints/DeepLab_Focal_test_run/best_model-epoch=0193-val_loss=0.08688327763229609.pt" \
    --Resnet_size 101 \