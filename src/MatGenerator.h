
/*
	define clarations for ParallelGen_RandomNonSymetricSparseMatrix_CSR_xx calls
*/
#ifndef MPI_HEADERS
#include <mpi.h>
#define MPI_HEADERS
#endif

#include "DataTypes.h"

#define ITEMS_PER_ROW_LOWER_BOUND 1
// SPARSITY_RATIO<=1.0
#define SPARSITY_RATIO 1.0
#define VAL_UPPER 1.0
#define VAL_LOWER 2.0
#define CHANGER_THRESHOLD 10

// #define DEV

#ifdef	__cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void TridiagonalCSR_val(csrType_local * mat, long rowDim, long colDim, double val);

// when both dim and offDiaHalfBandwidthRatio are small, 
//  nnz per row may be less than the lowest requirement, problem may happen

long ParallelGen_RandomNonSymetricSparseMatrix_CSR(csrType_local * localMat \
											,long dim, double offDiaHalfBandwidthRatio, int myid, int numprocs);

#ifdef	__cplusplus
}
#endif
