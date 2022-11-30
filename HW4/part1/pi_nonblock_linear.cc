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
    long long int global_count;
    float x, y;

    MPI_Barrier(MPI_COMM_WORLD);
    while (n--) {
	x = rand_r(&seed) / ((float)RAND_MAX);
	y = rand_r(&seed) / ((float)RAND_MAX);
	if(x * x + y * y <= 1.0) ++world_count;
    }

    MPI_Request req;
    if (world_rank > 0)
    {
        // TODO: MPI workers
	MPI_Isend(&world_count, 1, MPI_LONG_LONG_INT, 0, world_rank, MPI_COMM_WORLD, &req);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        // MPI_Request requests[];
	global_count = world_count;
	for (int i = 1; i < world_size; i++)
	{
		MPI_Irecv(&world_count, 1, MPI_LONG_LONG_INT, i, i, MPI_COMM_WORLD, &req);
		MPI_Wait(&req, &status);
		global_count += world_count;
	}
        //MPI_Waitall(world_size, req, status);
    }

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
