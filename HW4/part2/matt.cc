#include <iostream>
#include <mpi.h>
#include <unistd.h>

#define MASTER_WEIGHT 4

char buffer[40960];
int buffer_ptr = 0;
ssize_t buffer_end = 40960;

inline uint8_t get_value () {
    while (1) {
        if (buffer_ptr == buffer_end) {
            while (!(buffer_end = read(0, buffer, 40960))) ;
            buffer_ptr = 0;
        }
        if (buffer[buffer_ptr] >= '0')
            break;
        buffer_ptr++;
    }
    uint8_t tmp = 0;
    while (1) {
        if (buffer_ptr == buffer_end) {
            while (!(buffer_end = read(0, buffer, 40960))) ;
            buffer_ptr = 0;
        }
        if (buffer[buffer_ptr] < '0')
            break;
        tmp = tmp * 10 + buffer[buffer_ptr] - '0';
        buffer_ptr++;
    }
    return tmp;
}

inline uint16_t get_value_16 () {
    while (1) {
        if (buffer_ptr == buffer_end) {
            while (!(buffer_end = read(0, buffer, 40960))) ;
            buffer_ptr = 0;
        }
        if (buffer[buffer_ptr] >= '0')
            break;
        buffer_ptr++;
    }
    uint16_t tmp = 0;
    while (1) {
        if (buffer_ptr == buffer_end) {
            while (!(buffer_end = read(0, buffer, 40960))) ;
            buffer_ptr = 0;
        }
        if (buffer[buffer_ptr] < '0')
            break;
        tmp = tmp * 10 + buffer[buffer_ptr] - '0';
        buffer_ptr++;
    }
    return tmp;
}

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int matrix_size[3];

	if (world_rank) {
		MPI_Bcast(matrix_size, 3, MPI_INT, 0, MPI_COMM_WORLD);
		*n_ptr = (int)matrix_size[0];
		*m_ptr = (int)matrix_size[1];
		*l_ptr = (int)matrix_size[2];
	}
	else {
		matrix_size[0] = (int)get_value_16();
		matrix_size[1] = (int)get_value_16();
		matrix_size[2] = (int)get_value_16();
		*n_ptr = matrix_size[0];
		*m_ptr = matrix_size[1];
		*l_ptr = matrix_size[2];
		MPI_Bcast(matrix_size, 3, MPI_INT, 0, MPI_COMM_WORLD);
	}

	int matrix_a_size = (*n_ptr)*(*m_ptr);
	int matrix_b_size = (*m_ptr)*(*l_ptr);
    *a_mat_ptr = (int*)calloc(matrix_a_size, sizeof(int));
	*b_mat_ptr = (int*)calloc(matrix_b_size, sizeof(int));

	if (world_rank == 0)
	{
		for(int i = 0 ; i < matrix_a_size ; i++)
			*(*a_mat_ptr + i) = get_value();
		for (int i = 0; i < (*m_ptr); i++)
		{
			for (int j = 0; j < (*l_ptr); j++)
			*(*b_mat_ptr + i + j * (*m_ptr)) = get_value();
		}
	}

	MPI_Bcast(*a_mat_ptr, matrix_a_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(*b_mat_ptr, matrix_b_size, MPI_INT, 0, MPI_COMM_WORLD);	
}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int total_size = n * l;
	int *local_result = (int*) calloc(total_size, sizeof(int));
	int *result = (int*) calloc(total_size, sizeof(int));

	int portion = n / (world_size + MASTER_WEIGHT);
	int start_ptr, end_ptr;
	// distribute loading
	if (world_rank == 0)
	{
		start_ptr = portion * (world_size - 1);
		end_ptr = n;
	} else
	{
		start_ptr = portion * (world_rank - 1);
		end_ptr = start_ptr + portion;
	}
	// int start_ptr = world_rank * (n / world_size);
	// int end_ptr = (world_rank == world_size - 1) ? n : start_ptr + (n / world_size);

	int sum;
	int i = start_ptr, j, k;
	int mat_res_ptr = start_ptr * l, count = 0;

	for (; i < end_ptr; i++)
	{
		for (j = 0; j < l; j++)
		{
			sum = 0;
			for (k = 0; k < m; k++)
			{
				sum += a_mat[i * m + k] * b_mat[j * m + k];
			} 
			local_result[mat_res_ptr + count++] = sum;
		}
	}

	MPI_Reduce(local_result, result, total_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (!world_rank)
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < l; j++)
				printf("%d ", result[i * l + j]);
			putchar('\n');
		}
	}
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
    if (a_mat)
        delete [] a_mat;
    if (b_mat)
        delete [] b_mat;
}