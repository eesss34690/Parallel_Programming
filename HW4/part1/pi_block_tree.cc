#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status;
    unsigned int seed = time(NULL)* world_rank;
    long long int world_count = 0;
    long long int n = tosses / world_size;
    long long int global_count = world_count;
    float x, y;

    MPI_Barrier(MPI_COMM_WORLD);
    while (n--) {
	x = rand_r(&seed) / ((float)RAND_MAX);
	y = rand_r(&seed) / ((float)RAND_MAX);
	if(x * x + y * y <= 1.0) ++world_count;
    }

    // TODO: binary tree redunction
    if (world_size > 1)
    {
    	for (int i = 2; i <= world_size; i *= 2)
	{
	    if (world_rank % i == 0)
    	    {
        	// TODO: handle workers
		global_count = world_count;
		MPI_Recv(&world_count, 1, MPI_LONG_LONG_INT, world_rank + i / 2, 0, MPI_COMM_WORLD, &status);
		global_count += world_count;
		world_count = global_count;
    	    }
    	    else
    	    {
		MPI_Send(&world_count, 1, MPI_LONG_LONG_INT, world_rank - i / 2, 0, MPI_COMM_WORLD);
		break;
    	    }
	}
    }
    else
	    global_count = world_count;

    if (world_rank == 0)
    {
        // TODO: PI result
	pi_result = 4.0 * (double) global_count / tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
