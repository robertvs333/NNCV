wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.01 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "DeepLab_CrossEntropy" \
    --loss-function "cross_entropy"\
    --model-arch "deeplabv3plus" \
