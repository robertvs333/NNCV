wandb login
cd .. 
python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 150 \
    --lr 0.02 \
    --num-workers 12 \
    --seed 42 \
    --experiment-id "DeepLab_CEDICE_test_run" \
    --loss-function "ce_dice"\
    --model-arch "deeplabv3plus" \
    --resnet-size 101 \
    --freeze-backbone
