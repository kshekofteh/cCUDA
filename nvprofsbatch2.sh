#!/bin/bash -l

#SBATCH -p mantaro
#SBATCH -w atlas
#SBATCH --gres=gpu:2


srun -p mantaro -w atlas  --gres=gpu nvprof   ./_test_runtime_mm_vadd  -numElements=120000  -wA=512 -hA=1024 -wB=1024 -hB=512  -grid_rows=512  -grid_cols=512 -pyramid_height=2 -total_iterations=250 -ofile=output1.o -tfile=../rodinia/data/hotspot/temp_4096 -pfile=../rodinia/data/hotspot/power_4096   -bC=33554432  -obj -concurrent -simpl -rando
