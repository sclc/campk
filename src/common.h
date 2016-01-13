#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <queue>

#ifndef MPI_HEADERS
#include <mpi.h>
#define MPI_HEADERS
#endif

#ifndef GLOBAL_VARS
#include "globalvars.h"
#define GLOBAL_VARS
#endif 

#include "DataTypes.h"

#ifdef	__cplusplus
extern "C" {
#endif

char * concatStr (char * s1, char * s2);

void parseCSV (char* filename, double** output, long numRows, long numCols);

void Local_Dense_Mat_Generator(denseType * mat, long num_rows, long num_cols,\
	                 double ranMin, double ranMax);

bool RecursiveDependentEleComputation(std::map<long, std::vector<double> >& remoteValResultRecoder, \
                                      csrType_local_var mat, long  eleIdx, int my_level,\
                                      double *k_level_result, long myRowStart, long myRowEnd, long myNumRow,
                                      int myid, int numprocs);
bool RecursiveDependentEleComputation_mut(double * remoteValResultRecoderZone, \
                                      csrType_local_var mat, long  eleIdx, int my_level,\
                                      double *k_level_result, long myRowStart, long myRowEnd, long myNumRow,
                                      int myid, int numprocs);

bool RecursiveDependentEleComputation_mut2(double * remoteValResultRecoderZone, \
                                      csrType_local_var mat, long  eleIdx, int my_level,\
                                      double *k_level_result, long myRowStart, long myRowEnd, long myNumRow,
                                      int myid, int numprocs);

void dense_matrix_local_transpose_row_order(denseType inMat, denseType outMat);

//even if both src and target are denseType matrices,
// we actually copy consecutive element from src.data to target.data
void dense_entry_copy_disp(denseType src, long srcDispStart,denseType target, long tarDispStart, long count);

// res[output_mat_disp ...] = alpha*mat1 + beta*mat2, TP, third place
// note: one each processor, mat1 and mat2 should have the same shape
// res matrix get the result, however, a displacement should be promised
void dense_mat_mat_add_TP_targetDisp(denseType mat1, denseType mat2, denseType output_mat, long output_mat_disp
                                    ,double alpha, double beta, int myid, int numprocs);

// res[output_mat_disp ...] = alpha*mat1[disp1...] + beta*mat2[disp2...] + gamma*mat3[disp3...], TP, third place
// length data items will be processed
// note: one each processor, mat1 and mat2 and mat3 should have the same shape
// res matrix get the result, however, a displacement should be promised
void dense_array_mat_mat_mat_add_TP_disp(double* mat1Data, long mat1Disp
                                             , double* mat2Data, long mat2Disp
                                             , double* mat3Data, long mat3Disp
                                             , double* output_mat, long output_mat_disp
                                             , long length
                                             , double alpha, double beta, double gamma
                                             ,int myid, int numprocs);

void DenseMatrixComparsion (denseType mat1, denseType mat2);

void ClearQueue (std::queue<long> &rowIdxScanQueue);

void RecurisiveCallOverheadProfiling (int level);


#ifdef	__cplusplus
}
#endif



#define DB_RDEC

#ifndef ASSERTION_DEBUG
#define ASSERTION_DEBUG
#endif

// #define DB_RDEC_1_1
// #define DB_RDEC_1_2
// #define DB_RDEC_1_3
// #define DB_RDEC_1_4

