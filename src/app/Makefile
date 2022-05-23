NVCC = nvcc
CXX = g++
PROJ_DIR = ..
LIB_DIR = ${PROJ_DIR}/../lib

BLISS_DIR = ${LIB_DIR}/bliss-0.73
MGPU_DIR = ${LIB_DIR}/moderngpu
MPMCQUEUE_DIR= ${LIB_DIR}/MPMCQueue

INCLUDE = -I$(PROJ_DIR) -I${MGPU_DIR}/src -I${BLISS_DIR} -I${MPMCQUEUE_DIR}/include

MGPU_FLAGS = --expt-extended-lambda -use_fast_math -Wno-deprecated-declarations
OPEN_FLAGS = -DOPENMP -Xcompiler -fopenmp -lgomp -lpthread
BLISS_LIB = -L${BLISS_DIR}/bliss -lbliss
# To support backtrace, which is useful for debugging
BACKTRACE_FLAGS = -Xcompiler -rdynamic

CFLAGS += -std=c++17 \
          -DPRINT_INFO \
          -O3
#          -O2 -g
#          -DNVPROFILE \
#          -DDEBUG  \

# 3090
#ARCH = -arch=sm_86 -gencode=arch=compute_86,code=sm_86
# 2080ti and t4
ARCH = -arch=sm_75 -gencode=arch=compute_75,code=sm_75
# v100
#ARCH = -arch=sm_70 -gencode=arch=compute_70,code=sm_70
# 2080ti in 3090
#ARCH = -arch=sm_75 -gencode=arch=compute_75,code=sm_75 \
#                   -gencode=arch=compute_86,code=sm_86

all: pm_pipeline
build: pm_pipeline

pm_pipeline: pattern_matching_pipeline.cu
	$(NVCC) $(CFLAGS) ${OPEN_FLAGS} ${MGPU_FLAGS} ${ARCH} $(INCLUDE) -o $@ $^ ${BLISS_LIB}

clean:
	rm -rf pm_pipeline
