# Training and Evaluation on Snellius HPC

This guide details how to run your training and evaluation scripts on the Snellius High-Performance Computing (HPC) cluster using SLURM job scripts and Apptainer containers.

## Prerequisites

Before submitting any jobs, ensure you have completed the initial setup steps as outlined in `README-Slurm.md`:

1.  **Cloned your repository** on the cluster.
2.  **Downloaded the data and container** (`container.sif`) using `download_docker_and_data.sh`.
3.  **Configured your `.env` file** with your `WANDB_API_KEY` and `WANDB_DIR`.

No additional steps are needed compared with the already provided Readme files.

## How to Submit a Training Job

Training jobs are submitted using the `jobscript_slurm.sh` script, which in turn executes `main.sh` inside the Apptainer container.

To submit a job:

```bash
chmod +x jobscript_slurm.sh
sbatch jobscript_slurm.sh
```

SLURM will then queue and execute your job based on resource availability.

## Understanding `jobscript_slurm.sh`

The `jobscript_slurm.sh` file defines the resources your job will request from the SLURM scheduler. You can modify these parameters to suit your experiment's needs. For evaluation scripts the time limit may be reduced to 10 minutes to reduce the time for planning. For training set a limit of 4:00:00.
Here an example of jobscript_slurm.sh:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=00:10:00

srun apptainer exec --nv --env-file .env container.sif /bin/bash Evaluation_scripts/eval.sh
```
For training use the Training_scripts folders. Targuments to train.py are described here:
--experiment-id: Sets the name of the run in Weights & Biases (W&B) and is used to name the local folder where your .pt model checkpoints are saved.
--model-arch: Choose between "deeplabv3plus" and "unet".
--loss-function: Choose between "cross_entropy", "cross_entropy_weighted", "focal", and "ce_dice".
--freeze-backbone: A flag specific to DeepLabV3+ that freezes the ResNet backbone for the first 10 epochs before dynamically unfreezing it.
Make sure to activate this script using the jobscript_slurm.sh





``

## Monitoring and Debugging

-   **Monitor job status**: Use `squeue -u <your-username>` to see your jobs.
-   **Check logs**: SLURM creates output files (e.g., `slurm-<job_id>.out`) in your submission directory. These contain `print` statements and error messages.
-   **Weights & Biases**: Monitor training progress, loss curves, and validation predictions in real-time on your W&B dashboard.

## Local Evaluation

After training, you can evaluate your model locally on the cluster (or on your machine if you download the checkpoint) using `Evaluation_scripts/eval_local.py` together with the jobscript_slurm.sh. This script calculates mIoU and per-class IoU.

Example usage:

```bash
python Evaluation_scripts/eval_local.py \
    --data_dir ./data/cityscapes \
    --model_path ./checkpoints/deeplabv3plus-r101-focal/final_model-epoch=0074-val_loss=0.7890.pt \
    --model_arch deeplabv3plus \
    --Resnet_size 101
```

Remember to adjust `--model_path`, `--model_arch`, and `--Resnet_size` to match your trained model.

```
```
## Submission 
The details for submission are the same as in the other Readme's however in predict.py you need to make sure that you adjust ARCH to the correct architecture. The correct names are: "unet" or "deeplabv3plus".
```


