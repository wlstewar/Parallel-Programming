#!/bin/bash

####### select partition (check CCR documentation)
#SBATCH --partition=general-compute --qos=general-compute

####### set memory that nodes provide (check CCR documentation, e.g., 32GB)
#SBATCH --mem=32000

####### make sure no other jobs are assigned to your nodes


####### further customizations
#SBATCH --job-name="a0"
#SBATCH --output=%j.stdout
#SBATCH --error=%j.stderr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:02:30

for i in {1..10}
do
	echo "run $i:"
	./a0 2147483647
done
