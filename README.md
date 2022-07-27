VSGM
=======
VSGM: View-Based GPU-Accelerated Subgraph Matching on Large Graphs

Dependencies
-------------
1. nvidia/cuda:11.3.0
2. g++ - 7

Submodule
-------------
    $ git submodule init
    $ git submodule update

Build
-------------

	$ cd lib/bliss-0.73/bliss && make
    $ cd src/preprocess && make -j
    $ cd src/tools && make -j
    $ cd src/app && make -j

Change Data Format
-------------

    $ mkdir test
    $ cd test
    $ wget https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
    $ gunzip com-friendster.ungraph.txt.gz
    $ ../src/preprocess/preprocess -f com-friendster.ungraph.txt -d 0 -o 1  # build csr binary

K-Means Clustering
-------------

    $ cd test
    $ ../src/tools/features -f com-friendster.ungraph.txt.bin -h 1
    $ ../src/tools/kmeans -f com-friendster.ungraph.txt.bin.1hop.bin -i 10 -t 32 -k 4 -init 0 -rc 1  # i: iterations, t: threads, k: cluster num, init: 0 - kmeans++, 1 - random, rc: enable reorder cluster

View Bin Packing
-------------

    $ cd test
    $ cp com-friendster.ungraph.txt.bin.kmeans.4 com-friendster.ungraph.txt.bin.kmeans.1x4
    $ ../src/tools/view_packing -gf com-friendster.ungraph.txt.bin -pf com-friendster.ungraph.txt.bin.kmeans.1x4 -h 2 -m 10 -t 1 -s 1 -d 1  # for 2 hop
    $ ../src/tools/view_packing -gf com-friendster.ungraph.txt.bin -pf com-friendster.ungraph.txt.bin.kmeans.1x4 -h 3 -m 10 -t 1 -s 1 -d 1  # for 3 hop

Run Subgraph Matching
-------------

    $ cd test
    $ ../src/app/pm_pipeline -f com-friendster.ungraph.txt.bin -t 32 -m 10 -dm 1 -qs 3 -pn 1 -cn 1 -kc 3  # triangle counting
    $ ../src/app/pm_pipeline -f com-friendster.ungraph.txt.bin -t 32 -m 10 -dm 1 -qs 3 -pn 1 -cn 1 -p 2  # pattern 2

Acknowledgements
-------------
1. [GPU-Accelerated Subgraph Enumeration on Partitioned Graphs. SIGMOD 2020.](https://github.com/guowentian/SubgraphMatchGPU)
2. [Bliss Library](http://www.tcs.hut.fi/Software/bliss/)
3. [ModernGPU Library](https://github.com/moderngpu/moderngpu)
4. [MPMCQueue Library](https://github.com/rigtorp/MPMCQueue)
