#!/bin/bash -l

#SBATCH -p mantaro
#SBATCH -w atlas
#SBATCH --gres=gpu:2

nvprof  -o testrnd4_vadd_hs_tran_hist_01rnd.nvvp -f ./_test_vadd_hs_tran_hist  -numElements=10000     -grid_rows=512  -grid_cols=512 -pyramid_height=2 -total_iterations=250 -ofile=output1.o -tfile=../rodinia/data/hotspot/temp_4096 -pfile=../rodinia/data/hotspot/power_4096   -bC=33554432   -trnsp_size=4096  -obj  -random
nvprof  -o testrnd4_vadd_hs_tran_hist_01ser.nvvp -f ./_test_vadd_hs_tran_hist  -numElements=10000  -grid_rows=512  -grid_cols=512 -pyramid_height=2 -total_iterations=250 -ofile=output1.o -tfile=../rodinia/data/hotspot/temp_4096 -pfile=../rodinia/data/hotspot/power_4096   -bC=33554432   -trnsp_size=4096  -obj  -rando
nvprof  -o testrnd4_vadd_hs_tran_hist_01memcomp.nvvp -f ./_test_vadd_hs_tran_hist  -numElements=10000   -grid_rows=512  -grid_cols=512 -pyramid_height=2 -total_iterations=250 -ofile=output1.o -tfile=../rodinia/data/hotspot/temp_4096 -pfile=../rodinia/data/hotspot/power_4096   -bC=33554432   -trnsp_size=4096  -obj -memcomp 
