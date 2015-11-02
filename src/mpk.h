//
#ifndef MPI_HEADERS
#include <mpi.h>
#define MPI_HEADERS
#endif

#include "DataTypes.h"
#include "common.h"
#include "readMTX.h"
#include "matrixType.h"
#include "MatGenerator.h"

#ifdef	__cplusplus
extern "C" {
#endif

void Sparse_Csr_Matrix_Distribution(csrType_local * localMat, matInfo * mat_info,\
    	                            int myid, int numprocs, char* path, char* mtx_filename);

void spmm_csr_v1 (csrType_local csr_mat, denseType dense_mat, denseType *res_mat, double * global_swap_zone, int myid, int numprocs);

//double * global_swap_zone is deprecated

void spmm_csr_info_data_sep_CBCG (csrType_local csr_mat, denseType dense_mat_info, double * dataSrc, long dataDisp
        , denseType *res_mat, int myid, int numprocs);

void mpk_v1 (denseType S_mat,  csrType_local mat, denseType R, long s
                                , double s_alpha, double s_beta, int myid, int numprocs);

// transferring only necessary data
void mpk_v2 (denseType S_mat,  csrType_local mat, denseType R, long s
                                , double s_alpha, double s_beta, int myid, int numprocs);

void spmv_csr_v3(csrType_local smat, denseType dmat, denseType resMat, \
                 long *sendIdxBuf, int *bufSendingCount, int *bufSendingDispls, \
                 double *sendVecDataBuf,int *remoteVecCount, int* remoteVecPtr,\
                 double *remoteVecDataBuf, int sendCount, int recvCount,int myid, int numprocs);

int prepareRemoteVec_spmv(csrType_local csr_mat, int myid, int numprocs, \
            int *remoteVecCount, int *remoteVecPtr, long *remoteVecIndex);

#ifdef	__cplusplus
}
#endif