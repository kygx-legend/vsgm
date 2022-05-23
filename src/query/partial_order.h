#pragma once

#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bliss/graph.hh"
#include "bliss/utils.hh"
#include "common/meta.h"

// Copied from: https://github.com/pdclab/peregrine/blob/master/core/PO.cc.

// partial orders between two vertex
typedef std::vector<std::pair<uintV, uintV>> PartialOrderPairs;

class PartialOrder {
 public:
  static std::string OrderToString(const std::vector<uint32_t>& p) {
    std::string res;
    for (auto v : p)
      res += std::to_string(v);
    return res;
  }

  static void CompleteAutomorphisms(std::vector<std::vector<uint32_t>>& perm_group) {
    // multiplying std::vector<uint32_t>s is just function composition: (p1*p2)[i] = p1[p2[i]]
    std::vector<std::vector<uint32_t>> products;
    // for filtering duplicates
    std::unordered_set<std::string> dups;
    for (auto f : perm_group)
      dups.insert(OrderToString(f));

    for (auto k = perm_group.begin(); k != perm_group.end(); k++) {
      for (auto l = perm_group.begin(); l != perm_group.end(); l++) {
        std::vector<uint32_t> p1 = *k;
        std::vector<uint32_t> p2 = *l;

        std::vector<uint32_t> product;
        product.resize(p1.size());
        for (unsigned i = 0; i < product.size(); i++)
          product[i] = p1[p2[i]];

        // don't count duplicates
        if (dups.count(OrderToString(product)) == 0) {
          dups.insert(OrderToString(product));
          products.push_back(product);
        }
      }
    }

    for (auto p : products)
      perm_group.push_back(p);
  }

  static std::vector<std::vector<uintV>> GetAutomorphisms(bliss::Graph* bg, size_t vertex_count) {
    std::vector<std::vector<uintV>> result;
    bliss::Stats stats;
    bg->find_automorphisms(
        stats,
        [](void* param, const unsigned int size, const unsigned int* aut) {
          std::vector<uintV> result_aut;
          for (unsigned int i = 0; i < size; i++)
            result_aut.push_back(aut[i]);
          ((std::vector<std::vector<uintV>>*)param)->push_back(result_aut);
        },
        &result);

    uint32_t counter = 0;
    uint32_t lastSize = 0;
    while (result.size() != lastSize) {
      lastSize = result.size();
      CompleteAutomorphisms(result);
      counter++;
      if (counter > 100)
        break;
    }

    return result;
  }

  static std::map<uintV, std::set<uintV>> GetAEquivalenceClasses(const std::vector<std::vector<uintV>>& aut, size_t vertex_count) {
    std::map<uintV, std::set<uintV>> eclasses;
    for (size_t i = 0; i < vertex_count; i++) {
      std::set<uintV> eclass;
      for (auto&& perm : aut)
        eclass.insert(perm[i]);
      uintV rep = *std::min_element(eclass.cbegin(), eclass.cend());
      eclasses[rep].insert(eclass.cbegin(), eclass.cend());
    }
    return eclasses;
  }

  static PartialOrderPairs GetConditions(bliss::Graph* bg, size_t vertex_count) {
    std::vector<std::vector<uintV>> aut = GetAutomorphisms(bg, vertex_count);
#ifdef PRINT_DETAILS
    std::cout << "automorphisms:" << std::endl;
    for (int i = 0; i < aut.size(); i++) {
      std::cout << i << ": ";
      for (int j = 0; j < aut[i].size(); j++)
        std::cout << aut[i][j] << " ";
      std::cout << std::endl;
    }
#endif
    std::map<uintV, std::set<uintV>> eclasses = GetAEquivalenceClasses(aut, vertex_count);
#ifdef PRINT_INFO
    std::cout << "equivalence classes:" << std::endl;
    for (auto mit = eclasses.begin(); mit != eclasses.end(); ++mit) {
      std::cout << mit->first << ": ";
      for (uintV v : mit->second)
        std::cout << v << " ";
      std::cout << std::endl;
    }
#endif

    PartialOrderPairs result;
    auto eclass_it = std::find_if(eclasses.cbegin(), eclasses.cend(), [](auto&& e) { return e.second.size() > 1; });
    while (eclass_it != eclasses.cend() && eclass_it->second.size() > 1) {
      const auto& eclass = eclass_it->second;
      uintV n0 = *eclass.cbegin();

      for (auto&& perm : aut) {
        uintV min = *std::min_element(std::next(eclass.cbegin()), eclass.cend(), [perm](uint32_t n, uint32_t m) { return perm[n] < perm[m]; });
        result.emplace_back(n0, min);
      }
      aut.erase(std::remove_if(aut.begin(), aut.end(), [n0](auto&& perm) { return perm[n0] != n0; }), aut.end());

      eclasses = GetAEquivalenceClasses(aut, vertex_count);
      eclass_it = std::find_if(eclasses.cbegin(), eclasses.cend(), [](auto&& e) { return e.second.size() > 1; });
    }

    // remove duplicate conditions
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());

#ifdef PRINT_INFO
    std::cout << "conditions:" << std::endl;
    for (int i = 0; i < result.size(); i++)
      std::cout << result[i].first << " " << result[i].second << std::endl;
#endif
    return result;
  }
};
