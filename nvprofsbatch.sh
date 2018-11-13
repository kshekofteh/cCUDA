#!/bin/bash -l

#SBATCH -p mantaro
#SBATCH -w victoria
#SBATCH --gres=gpu:2

nvprof -o t05ser_vic.nvvp -f  ./_test_runtime  -numElements=160000  -wA=256 -hA=1024 -wB=1024 -hB=256 -C1=262144 -grid_rows=2048 -grid_cols=1024 -pyramid_height=2 -total_iterations=250 -ofile=output1.o -tfile=../rodinia/data/hotspot/temp_4096 -pfile=../rodinia/data/hotspot/power_4096  -trnsp_size=8192   -bC=1048576 -obj -concurren -simpl -rando -serial
nvprof -o t05_vic.nvvp -f  ./_test_runtime  -numElements=160000  -wA=256 -hA=1024 -wB=1024 -hB=256 -C1=262144 -grid_rows=2048 -grid_cols=1024 -pyramid_height=2 -total_iterations=250 -ofile=output1.o -tfile=../rodinia/data/hotspot/temp_4096 -pfile=../rodinia/data/hotspot/power_4096  -trnsp_size=8192   -bC=1048576 -obj -concurrent -simpl -rando -seria
