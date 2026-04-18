wandb login
cd .. 
python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.04 \
    --num-workers 12 \
    --seed 42 \
    --experiment-id "DeepLab_50_Focal_lr" \
    --loss-function "focal"\
    --model-arch "deeplabv3plus" \
    --resnet-size 50 \
    --freeze-backbone 
