#
# Copyright 2011-2015 NVIDIA Corporation. All rights reserved
# 
INCLUDES = -I /opt/cuda-8.0/extras/CUPTI/include,/share/kshekofteh/codes/nvidia/common/inc,common,.
EXE = _test_runtime_vec_hs_device

ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):../../libWin32:../../libx64
    LIBS= -lcuda -L ../../libWin32 -L ../../libx64 -lcupti
    OBJ = obj
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):../../lib
        LIBS= -Xlinker -framework -Xlinker cuda -L ../../lib -lcupti
    else
        export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):../../lib:/opt/cuda-8.0/extras/CUPTI/lib64
        LIBS= -lcuda -L ../../lib -L /opt/cuda-8.0/extras/CUPTI/lib64 -lcupti
    endif
    OBJ = o
endif

_test_runtime: _test_runtime.$(OBJ)
	nvcc -o $(EXE) _test_runtime.$(OBJ) $(LIBS)   -Wno-deprecated-gpu-targets

_test_runtime.$(OBJ): _test_runtime.cu
	nvcc  -Wno-deprecated-gpu-targets -c $(INCLUDES) $<

run: $(EXE)
	./$<

clean:
	rm -f $(EXE) _test_runtime.$(OBJ)
