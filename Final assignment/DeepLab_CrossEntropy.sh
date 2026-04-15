wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.02 \
    --num-workers 24 \
    --seed 42 \
    --experiment-id "DeepLab_CrossEntropy_lr" \
    --loss-function "cross_entropy"\
    --model-arch "deeplabv3plus" \
    --resnet-size 101 \
    --freeze-backbone \