wandb login
cd .. 
python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 200 \
    --lr 0.04 \
    --num-workers 12 \
    --seed 42 \
    --experiment-id "DeepLab_Focal_test_run" \
    --loss-function "focal"\
    --model-arch "deeplabv3plus" \
    --resnet-size 101 \
    --freeze-backbone
