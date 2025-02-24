#!/bin/bash
#SBATCH --job-name="DP-Exp" 	  # a name for your job
#SBATCH --partition=peregrine-gpu		  # partition to which job should be submitted
#SBATCH --qos=gpu_short					  # qos type
#SBATCH --nodes=1                		  # node count
#SBATCH --ntasks=1               		  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        		  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=30G         				  # total memory per node
#SBATCH --gres=gpu:3090:1			  # Request 1 GPU
#SBATCH --time=3:00:00          		  # total run time limit (HH:MM:SS)

module purge
module load python/anaconda

port=15237
ssh -N -f -R $port:localhost:$port falcon

jupyter-notebook --no-browser --port=$port







