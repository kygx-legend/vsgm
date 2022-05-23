GCC = g++
PROJ_DIR = ..
INCLUDE = $(PROJ_DIR) 
LIBS = -lm 
OPENMP_FLAGS = -DOPENMP -fopenmp -lgomp
CFLAGS += -std=c++11 -O3 
#CFLAGS += -std=c++11 -O2 -g 

all: preprocess_graph 
build: preprocess_graph 

OBJ = $(patsubst %.cc,%.o,$(wildcard *.cc))

%.o: %.cc
	$(GCC) $< $(CFLAGS) $(INCLUDE) -c -o $@
	
preprocess_graph: preprocess_graph.cc
	$(GCC) $(OPENMP_FLAGS) $(CFLAGS) -I$(INCLUDE) -o $@ $^ $(LIBS)

clean:
	rm -rf preprocess_graph
