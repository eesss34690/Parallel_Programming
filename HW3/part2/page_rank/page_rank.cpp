#include "page_rank.h"
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  double sumNoOut = 0.0;
  #pragma omp parallel for reduction(+:sumNoOut)
  for (int i = 0; i < numNodes; ++i)
  {
    	solution[i] = equal_prob;
    	if (outgoing_size(g, i) == 0)
	   sumNoOut += equal_prob;
  }

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
   double *score_new = (double*) calloc(numNodes, sizeof(double));
   double diff;

   while (1)
   {
	#pragma omp parallel for
	for (int i = 0 ; i < numNodes ; ++i)
	{
		double temp = 0.0;
		const Vertex *start = incoming_begin(g, i);
		const Vertex *end = incoming_end(g, i);
		for (const Vertex *v = start; v != end; v++)
			temp += solution[*v] / outgoing_size(g, *v);
		temp = (damping * temp) + (1.0 - damping) / numNodes;
		score_new[i] = temp + damping * sumNoOut / numNodes;
	}

	diff = 0.0, sumNoOut = 0.0;
	#pragma omp parallel for reduction(+:diff, sumNoOut)
	for (int i = 0; i < numNodes; i++)
	{
		diff += fabs(score_new[i] - solution[i]);
		solution[i] = score_new[i];
		if (!outgoing_size(g, i))
			sumNoOut += score_new[i];
	}
	if (diff < convergence)
		break;
   }

}
