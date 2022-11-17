#include <stdlib.h>
#include <pthread.h>
#include <cstdio>
#include <cstdint>
long long int numOfIterationperThread, in;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

// thread calculation of pi
static inline void* pi(void* arg)
{
	unsigned int seed = (uint64_t)arg;
	long long int localIn;
	for (long long int i = 0; i < numOfIterationperThread; i++)
	{
		double x = 2.0 * rand_r(&seed) / (RAND_MAX + 1.0) - 1.0;
		double y = 2.0 * rand_r(&seed) / (RAND_MAX + 1.0) - 1.0;
		double dist = x * x + y * y;
		if (dist <= 1.0) 
			localIn++;
	}
	// get ht elockto write count
	pthread_mutex_lock(&lock);
	in += localIn;
	pthread_mutex_unlock(&lock);
	// end
	pthread_exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[])
{
	// get the thread count and iteration count from argument
	long long int numOfIteration, numOfThread, i;
	numOfThread = atoll(argv[1]);
	numOfIteration = atoll(argv[2]);
	// create thread and count the iteration for one thread
	pthread_t threads[numOfThread];
	numOfIterationperThread = numOfIteration / numOfThread;
	// create threads
	for(i = numOfThread - 1; i >= 0; --i)
		pthread_create(&threads[i], NULL, pi, (void*)i);
	// collect threads
	for(i = numOfThread - 1; i>= 0; --i)
		pthread_join(threads[i], NULL);
	// print result
	printf("%.6f\n", 4.0 * (double)in / numOfIteration);
	return 0;
}
