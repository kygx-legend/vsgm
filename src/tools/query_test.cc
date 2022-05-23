#include "query/query.h"

#include <iostream>
#include <string>

#include "common/command_line.h"
#include "query/clique.h"
#include "query/graphlet.h"
#include "query/pattern.h"
#include "query/plan.h"

int main(int argc, char** argv) {
  if (argc == 1) {
    return -1;
  }

  // parse command line
  CommandLine cmd(argc, argv);
  int pattern_type = cmd.GetOptionIntValue("-p", P1);
  int k_c = cmd.GetOptionIntValue("-kc", 0);
  int k_m = cmd.GetOptionIntValue("-km", 0);

  if (k_c == 0 && k_m == 0) {
    Query* query = new Pattern((PresetPatternType)pattern_type);
    query->Print();
    Plan* plan = new Plan(query);
    plan->Optimize();
    plan->Print();
    return 0;
  }

  if (k_c > 0) {
    Query* query = new Clique(k_c);
    query->Print();
    Plan* plan = new Plan(query);
    plan->Optimize();
    plan->Print();
    return 0;
  }

  if (k_m > 0) {
    Graphlet* graphlet = new Graphlet(k_m);
    auto& queries = graphlet->GetQueries();
    std::vector<Plan*> plans;
    std::vector<int> hops;
    int hop = 0;
    for (int i = 0; i < queries.size(); i++) {
      queries[i]->Print();
      Plan* plan = new Plan(queries[i]);
      plan->Optimize();
      plan->Print();
      plans.push_back(plan);
      hops.push_back(plan->GetHop());
      if (plan->GetHop() > hop)
        hop = plan->GetHop();
    }

    std::cout << "-----------" << std::endl;
    for (int i = 0; i < hops.size(); i++)
      std::cout << i << ": " << hops[i] << std::endl;
    std::cout << "largest hop: " << hop << std::endl;
  }

  return 0;
}
