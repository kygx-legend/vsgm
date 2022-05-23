#include <cfloat>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "common/command_line.h"
#include "common/meta.h"
#include "common/time_measurer.h"
#include "graph/cpu_graph.h"

long unfind(std::vector<long>& un_father, long u) {
  return un_father[u] == u ? u : (un_father[u] = unfind(un_father, un_father[u]));
}

void ununion(std::vector<long>& un_father, long u, long v) {
  un_father[u] = v;
}

int main(int argc, char* argv[]) {
  if (argc == 1) {
    return -1;
  }

  CommandLine cmd(argc, argv);
  std::string in_filename = cmd.GetOptionValue("-f", "./data/com-dblp.ungraph.txt");
  int part_num = cmd.GetOptionIntValue("-p", 4);
  int hop = cmd.GetOptionIntValue("-h", 1);

  Graph* graph = new Graph(in_filename, false);

  TimeMeasurer timer;
  timer.StartTimer();

  size_t vertex_num = graph->GetVertexCount();
  size_t edge_num = graph->GetEdgeCount() / 2;

  uintE* row_ptrs = graph->GetRowPtrs();
  uintV* cols = graph->GetCols();

  float gamma = 1.5;
  float alpha = sqrt(part_num) * edge_num / pow(vertex_num, gamma);
  float v = 1.1;
  float miu = v * vertex_num / part_num;

  std::vector<long> part_info = std::vector<long>(vertex_num, 0);
  std::vector<long> vertex_num_part = std::vector<long>(part_num, 0);

  std::vector<long> un_father(vertex_num);
  std::vector<long> un_first(part_num, -1);

  for (long i = 0; i < vertex_num; ++i)
    un_father[i] = i;

  for (uintV v = 0; v < vertex_num; v++) {
    if (v % 100000 == 0)
      std::cout << v << std::endl;

    float max_score = -FLT_MAX;
    int max_part = 0;
    for (long i = 0; i < part_num; i++) {
      if (vertex_num_part[i] <= miu) {
        double delta_c = alpha * (pow(vertex_num_part[i] + 1, gamma) - pow(vertex_num_part[i], gamma));
        float score = 0;
        for (long j = row_ptrs[v]; j < row_ptrs[v + 1]; j++) {
          uintV nid = cols[j];
          // this partition contains nid
          if (un_first[i] >= 0 && unfind(un_father, un_first[i]) == unfind(un_father, nid))
            score += 1;

          if (hop == 2) {
            for (long k = row_ptrs[nid]; k < row_ptrs[nid + 1]; k++) {
              uintV nnid = cols[k];
              if (un_first[i] >= 0 && unfind(un_father, un_first[i]) == unfind(un_father, nnid))
                score += 1;
            }
          }
        }
        score = score - delta_c;
        if (max_score < score) {
          max_score = score;
          max_part = i;
        }
      }
    }
    if (un_first[max_part] < 0)
      un_first[max_part] = v;
    ununion(un_father, v, un_first[max_part]);
    vertex_num_part[max_part] += 1;
    part_info[v] = max_part;
  }

  timer.EndTimer();
  timer.PrintElapsedMicroSeconds("fennel");

  std::string out_filename = in_filename + ".fennel.part." + std::to_string(part_num);
  std::ofstream outfile(out_filename.c_str());
  for (uintV v = 0; v < vertex_num; v++)
    outfile << part_info[v] << std::endl;
  outfile.close();

  return 0;
}
