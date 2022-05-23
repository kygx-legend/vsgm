GCC = g++
PROJ_DIR = ..
LIB_DIR = ${PROJ_DIR}/../lib
ROARING_DIR = ${LIB_DIR}/CRoaring

INCLUDE = $(PROJ_DIR)

LIBS = -lm 

OPENMP_FLAGS = -DOPENMP -fopenmp -lgomp
CFLAGS += -std=c++17 -O3
#CFLAGS += -std=c++17 -O2 -g
ACC_FLAGS = -funroll-loops

all: view_packing fennel kmeans features 
build: view_packing fennel kmeans features 

OBJ = $(patsubst %.cc,%.o,$(wildcard *.cc))

%.o: %.cc
	$(GCC) $< $(CFLAGS) $(INCLUDE) -c -o $@
	
view_packing: view_packing.cc
	$(GCC) $(OPENMP_FLAGS) $(CFLAGS) -I$(INCLUDE) -o $@ $^ $(LIBS)

fennel: fennel.cc
	$(GCC) $(OPENMP_FLAGS) $(CFLAGS) -I$(INCLUDE) -o $@ $^ $(LIBS)

kmeans: kmeans.cc
	$(GCC) $(OPENMP_FLAGS) $(CFLAGS) $(ACC_FLAGS) -I$(INCLUDE) -o $@ $^ $(LIBS)

features: features.cc
	$(GCC) $(OPENMP_FLAGS) $(CFLAGS) -I$(INCLUDE) -o $@ $^ $(LIBS)

query_test: query_test.cc
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ $^ ${BLISS_LIB}

clean:
	rm -rf view_packing fennel kmeans features
