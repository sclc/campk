// mpk header file
#include "DataTypes.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
// #include <mpi.h>
#include <queue>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include "DataTypes.h"

#ifndef MPI_HEADERS
#include <mpi.h>
#define MPI_HEADERS
#endif

#ifndef GLOBAL_VARS
#include "globalvars.h"
#define GLOBAL_VARS
#endif 

#ifdef	__cplusplus
extern "C" {
#endif

#include "common.h"
#include "matrixType.h"
#include "readMTX.h"

void Sparse_Csr_Matrix_Distribution_kLevel_net_v1(csrType_local * localMat, int k, matInfo * mat_info \
                                  , int myid, int numprocs,char* path, char* mtx_filename);

void campk_v2(denseType X, denseType &AkX , int kval \
	             , matInfo * mat_info, int myid, int numprocs \
	             , char* path, char* mtx_filename);

void campk_local_dependecy_BFS_v1 (csrType_local entireCsrMat, std::map<long, std::vector<int> > &dependencyRecoder \
                                 , long myNumRow, long myRowStart, long myRowEnd, short * k_level_locally_computable_flags, int kval \
                                 , int myid, int numprocs);

void campk_local_dependecy_BFS_v2 (csrType_local entireCsrMat, std::map<long, std::vector<int> > &dependencyRecoder \
                                 , long myNumRow, long myRowStart, long myRowEnd, short * k_level_locally_computable_flags, int kval \
                                 , int myid, int numprocs);

void campk_local_dependecy_BFS_v3 (csrType_local entireCsrMat, std::map<long, std::vector<int> > &dependencyRecoder \
                                 , long myNumRow, long myRowStart, long myRowEnd, short * k_level_locally_computable_flags, int kval \
                                 , int myid, int numprocs);

void campk_compacting_csr_v1 (csrType_local_var &compactedCSR, std::map<long, std::vector<int> > dependencyRecoder \
                           , csrType_local entireCsrMat, int myid, int numprocs);


int campk_comm_overlaping_local_computation_v1 (csrType_local_var compactedCSR, std::map<long, std::vector<int> > dependencyRecoder \
                                               , double *& buffer_vec_remote_recv, short  *k_level_locally_computable_flags \
                                               , double *k_level_result, long vec_result_length, long myNumRow, long myRowStart, long myRowEnd \
                                               , long averageNumRowPerProc, long *& vec_remote_recv_idx, int myid, int numprocs);

int campk_comm_overlaping_local_computation_v2 (csrType_local_var compactedCSR, std::map<long, std::vector<int> > dependencyRecoder \
                                               , double *& buffer_vec_remote_recv, short  *k_level_locally_computable_flags \
                                               , double *k_level_result, long vec_result_length, long myNumRow, long myRowStart, long myRowEnd \
                                               , long averageNumRowPerProc, long *& vec_remote_recv_idx, int myid, int numprocs);


void campk_after_comm_computation_original (csrType_local_var compactedCSR, double *k_level_result, short  *k_level_locally_computable_flags 
                                 , long vec_result_length, long myNumRow, long myRowStart, long myRowEnd,long *vec_remote_recv_idx 
				 , double * remoteValResultRecoderZone, std::vector< std::map <int,long> > levelPatternRemoteVals
				 , long offsetRemoteValCounter[], long offsetRemoteVal[] 
                                 , double * buffer_vec_remote_recv, int numRemoteVec, int kval, int myid, int numprocs);

void campk_after_comm_computation_openacc_v1 (csrType_local_var compactedCSR, double *k_level_result, short  *k_level_locally_computable_flags 
                                 , long vec_result_length, long myNumRow, long myRowStart, long myRowEnd,long *vec_remote_recv_idx 
				 , double * remoteValResultRecoderZone, std::vector< std::map <int,long> > levelPatternRemoteVals
				 , long offsetRemoteValCounter[], long offsetRemoteVal[] 
                                 , double * buffer_vec_remote_recv, int numRemoteVec, int kval, int myid, int numprocs);
#ifdef	__cplusplus
}
#endif

// #define DB_CAMPK_V1_1
#define CAMPK_PROF_ALL
#define CAMPK_PROF_DEP_BFS
#define CAMPK_PROF_COMPACT
#define CAMPK_PROF_COMM_COMPUT
#define CAMPK_PROF_AFTER_COMM_COMPUT

//#define CAMPK_PROF_AFTER_COMM_COMPUT_FUNC_1
//#define CAMPK_PROF_AFTER_COMM_COMPUT_FUNC_2
// #define CAMPK_PROF_AFTER_COMM_COMPUT_FUNC_V2_ALL
// #define CAMPK_PROF_AFTER_COMM_COMPUT_FUNC_V2_1
// #define CAMPK_PROF_AFTER_COMM_COMPUT_FUNC_OMP_1
// #define CAMPK_PROF_AFTER_COMM_COMPUT_FUNC_V2_DUP_CHECK

//#define PRINT_DEPENDENCY_RECODER

//#define DB_OPACC__1
#define DB_OPACC__2

