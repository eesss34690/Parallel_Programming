#include "bfs.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

using namespace std;

struct vertexSet {
  vertexSet(int n) {
    mask.resize(n);
    count = 0;
  }
  vector<uint8_t> mask;
  int count;
};

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertexSet *frontier,
    vertexSet *new_frontier,
    int *distances)
{
  int sum = 0;
  #pragma omp parallel for reduction(+:sum) schedule(guided, 1024)
  for (int i = 0; i < g->num_nodes; ++i) {
    if (frontier->mask[i]) {
      int start_edge = g->outgoing_starts[i];
      int end_edge = (i == g->num_nodes - 1)
                             ? g->num_edges
                             : g->outgoing_starts[i + 1];
      for (int j = start_edge; j < end_edge; j++) {
        int outgoing = g->outgoing_edges[j];
        if (distances[outgoing] == NOT_VISITED_MARKER) {
          distances[outgoing] = distances[i] + 1;
          new_frontier->mask[outgoing] = 1;
          ++sum;
        }
      }
    }
  }
  new_frontier->count = sum;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol) {
  vertexSet list1(graph->num_nodes);
  vertexSet list2(graph->num_nodes);

  vertexSet *frontier = &list1;
  vertexSet *new_frontier = &list2;

  // initialize all nodes to NOT_VISITED
  #pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  list1.mask[ROOT_NODE_ID] = 1;
  list1.count = 1;
  sol->distances[ROOT_NODE_ID] = 0;

  while (list1.count != 0) {
    list2.count = 0;
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
    top_down_step(graph, frontier, new_frontier, sol->distances);
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
    swap(frontier, new_frontier);
    fill(new_frontier->mask.begin(), new_frontier->mask.end(), 0);
  }
}

// Implements buttom-up BFS.
// TODO: Wrong for now
void bottom_up_step(
    Graph g,
    vertexSet *frontier,
    vertexSet *new_frontier,
    int *distances)
{
  int sum = 0;
  #pragma omp parallel for reduction(+:sum) schedule(guided, 1024)
  for (int i = 0; i < g->num_nodes; ++i) {
    if (distances[i] == NOT_VISITED_MARKER) {
      int start_edge = g->incoming_starts[i];
      int end_edge = (i == g->num_nodes - 1)
                             ? g->num_edges
                             : g->incoming_starts[i + 1];
      // int end_edge = start_edge + incoming_size(g, i);
      #pragma omp private(j)
      for (; start_edge < end_edge; ++start_edge) {
        int incoming = g->incoming_edges[start_edge];
        if (frontier->mask[incoming]) {
          distances[i] = distances[incoming] + 1;
          new_frontier->mask[i] = 1;
          break;
        }
      }
      sum = sum + (start_edge != end_edge);
    }
  }
  new_frontier->count = sum;
}

void bfs_bottom_up(Graph graph, solution *sol) {
  // For PP students:
  //
  // You will need to implement the "bottom up" BFS here as
  // described in the handout.
  //
  // As a result of your code's execution, sol.distances should be
  // correctly populated for all nodes in the graph.
  //
  // As was done in the top-down case, you may wish to organize your
  // code by creating subroutine bottom_up_step() that is called in
  // each step of the BFS process.
  vertexSet list1(graph->num_nodes);
  vertexSet list2(graph->num_nodes);

  vertexSet *frontier = &list1;
  vertexSet *new_frontier = &list2;

  // initialize all nodes to NOT_VISITED
  #pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  list1.mask[ROOT_NODE_ID] = 1;
  list1.count = 1;
  sol->distances[ROOT_NODE_ID] = 0;

  while (list1.count != 0) {
    list2.count = 0;
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
    bottom_up_step(graph, frontier, new_frontier, sol->distances);
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
    swap(frontier, new_frontier);
    fill(new_frontier->mask.begin(), new_frontier->mask.end(), 0);
  }
}

void bfs_hybrid(Graph graph, solution *sol) {
  // For PP students:
  //
  // You will need to implement the "hybrid" BFS here as
  // described in the handout.
  int numNodes = graph->num_nodes;
  vertexSet list1(numNodes);
  vertexSet list2(numNodes);

  vertexSet *frontier = &list1;
  vertexSet *new_frontier = &list2;
  
  // initialize all nodes to NOT_VISITED
  #pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  list1.mask[ROOT_NODE_ID] = 1;
  list1.count = 1;
  sol->distances[ROOT_NODE_ID] = 0;

  while (list1.count != 0) {
    list2.count = 0;
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
    if (list1.count >= numNodes / 100)
      bottom_up_step(graph, frontier, new_frontier, sol->distances);
    else
      top_down_step(graph, frontier, new_frontier, sol->distances);
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
    swap(frontier, new_frontier);
    fill(new_frontier->mask.begin(), new_frontier->mask.end(), 0);
  }
}
