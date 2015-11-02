#include "mpk.h"

//
void Sparse_Csr_Matrix_Distribution(csrType_local * localMat, matInfo * mat_info \
                                  , int myid, int numprocs \
                                  , char* path, char* mtx_filename) {

    // use MPI blocking communication
    cooType globalMat;

    csrType_local entireCsrMat;


    int * col_local_start_idx = (int *) calloc(numprocs, sizeof (int));
    // long * col_local_start_idx = (long *) calloc(numprocs, sizeof (long));
    // size_t * col_local_start_idx = (size_t *) calloc(numprocs, sizeof (size_t));

    int * col_local_length = (int *) calloc(numprocs, sizeof (int));
    // long * col_local_length = (long *) calloc(numprocs, sizeof (long));
    // size_t * col_local_length = (size_t *) calloc(numprocs, sizeof (size_t));

    // long * val_local_start_idx = (long *) calloc(numprocs, sizeof (long));
    // long * val_local_length = (long *) calloc(numprocs, sizeof (long));
    int * val_local_start_idx = (int *) calloc(numprocs, sizeof (int));
    int * val_local_length = (int *) calloc(numprocs, sizeof (int));

    long rows_per_proc, num_total, num_row_local;

    long idx;
    int ierr;

    long num_row_col[2];


    // printf("long myid is %ld\n", myid);
    if (myid == 0) {
        // rank 0 read mtr

        readMtx_info_and_coo(path, mtx_filename, mat_info, & globalMat);
        // printf("so far so good C\n");
        // exit(0);

        printf("there are %ld elements in matrix \n", mat_info->nnz);
#ifdef DEBUG_DATA_DIST
        //check_coo_matrix_print(globalMat, *mat_info);
#endif
        Converter_Coo2Csr(globalMat, &entireCsrMat, mat_info);
#ifdef DEBUG_DATA_DIST
        //check_csr_matrix_print(entireCsrMat);
#endif
        num_row_col[0] = mat_info->num_rows;
        num_row_col[1] = mat_info->num_cols;

        delete_cooType(globalMat);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast((void *) num_row_col, 2, MPI_LONG, 0, MPI_COMM_WORLD);
    num_total = num_row_col[0];
    rows_per_proc = num_total / numprocs;


    if (myid != 0) {
        entireCsrMat.row_start = (long *) calloc(num_total + 1, sizeof (long));
    }
#ifdef DEBUG_DATA_DIST
    printf("myid=%d, num_row_col[0]=%ld, num_row_col[1]=%ld \n"
            , myid, num_row_col[0], num_row_col[1]);
#endif


    if (myid == (numprocs - 1)) {
        num_row_local = num_total - rows_per_proc * (numprocs - 1);
    } else {
        num_row_local = rows_per_proc;
    }

    localMat->num_rows = num_row_local;

    localMat->num_cols = num_row_col[1];

    localMat->row_start = (long *) calloc(num_row_local + 1, sizeof (long));
    localMat->start = myid * rows_per_proc;
#ifdef DEBUG_DATA_DIST
    //    printf("myid=%d, localMat->start=%d, localMat->num_rows=%d\n", myid, localMat->start, localMat->num_rows);
#endif    

    ierr = MPI_Bcast((void*) entireCsrMat.row_start, num_total + 1, MPI_LONG
            , 0, MPI_COMM_WORLD);
#ifdef DEBUG_DATA_DIST
    printf("myid=%d, ", myid);
    for (idx = 0; idx < num_total + 1; idx++) {
        printf("%d, ", entireCsrMat.row_start[idx]);
    }
    printf("\n");
#endif


    // for local_car is only for local data, we need to shift row_start value
    // , or say, the first value of row_start should be 0 for all processes
    for (idx = 0; idx <= num_row_local; idx++) {
        localMat->row_start[idx] = entireCsrMat.row_start[myid * rows_per_proc + idx]
                - entireCsrMat.row_start[myid * rows_per_proc];
    }

#ifdef DEBUG_DATA_DIST_LOCAL_ROW_START
    printf("##################local row_start test myid=%d, \t", myid);
    for (idx = 0; idx <= num_row_local; idx++) {
        printf("%d, ", localMat->row_start[idx]);
    }
    printf("\n");
#endif



    for (idx = 0; idx < numprocs; idx++) {
        long row_start = idx * rows_per_proc;
        long row_end = row_start + rows_per_proc;
        if (idx == (numprocs - 1))
            row_end = num_total;

        //col_local_start_idx[idx] is int, check entireCsrMat.row_start[row_start] value range
        // col_local_start_idx[idx] = (int)entireCsrMat.row_start[row_start];
        col_local_start_idx[idx] = (int)entireCsrMat.row_start[row_start];
        // col_local_start_idx[idx] = (size_t)entireCsrMat.row_start[row_start];

        col_local_length[idx] = (int)entireCsrMat.row_start[row_end]
                - entireCsrMat.row_start[row_start];

        val_local_start_idx[idx] = (int)entireCsrMat.row_start[row_start];
        val_local_length[idx] = (int)(entireCsrMat.row_start[row_end]
                - entireCsrMat.row_start[row_start]);
    }
 
#ifdef DEBUG_DATA_DIST
    long total_entries = 0;
    for (idx = 0; idx < numprocs; idx++)
        total_entries += col_local_length[idx];
    printf("#########myid=%d, total_entries=%ld\n", myid, total_entries);
    printf("#########myid=%d, local_entries=%ld\n", myid, col_local_length[myid]);
    printf("#########myid=%d, val_local_length=%ld\n", myid, val_local_length[myid]);

#endif


    localMat->col_idx = (long *) calloc(col_local_length[myid], sizeof (long));
    localMat->csrdata = (double*) calloc(val_local_length[myid], sizeof (double));

    // printf("before MPI_Scatterv\n");
    // printf ("entireCsrMat.col_idx[3]: %ld\n", entireCsrMat.col_idx[3]);

    // col index scatterv
    ierr = MPI_Scatterv((void *) entireCsrMat.col_idx, (int*)col_local_length
            , (int*)col_local_start_idx, MPI_LONG
            , (void *) localMat->col_idx, (int)col_local_length[myid]
            , MPI_LONG, 0, MPI_COMM_WORLD);

    // printf("after MPI_Scatterv\n");
    // printf ("localMat->col_idx[3]: %ld\n", localMat->col_idx[3]);
    // printf("size of size_t, int, long, double: %d, %d, %d, %d\n"\
    //      , sizeof(size_t), sizeof(int), sizeof(long), sizeof(double));
    // exit(0);

    // val scatterv
    // int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
                 // MPI_Datatype sendtype, void *recvbuf, int recvcount,
                 // MPI_Datatype recvtype,
                 // int root, MPI_Comm comm)
    // runtime data wrongly distributed, if use long type displs
    ierr = MPI_Scatterv((void *) entireCsrMat.csrdata, (int*)val_local_length
            , (int*)val_local_start_idx, MPI_DOUBLE
            , (void *) localMat->csrdata, (int)val_local_length[myid]
            , MPI_DOUBLE, 0, MPI_COMM_WORLD);

    localMat->nnz = val_local_length[myid];
    // printf ("localMat->nnz : %ld\n", val_local_length[myid]);

#ifdef DEBUG_DATA_DIST_2
    // how to check this scatterv has been done correctly ?
    //, use small spd and check manually
    long temp_idx;
    printf("in SparseMatrixDistribution.c, myid=%d\t\t", myid);
    //    for (temp_idx=0; temp_idx<col_local_length[myid]; temp_idx++){
    //        printf( "%f ",localMat->csrdata[temp_idx]);
    // }
    for (temp_idx = 0; temp_idx < col_local_length[myid]; temp_idx++) {
        printf("%d ", localMat->col_idx[temp_idx]);
    }
    printf("distributed data print ending\n");
#endif
    // finally gabage management

    free(entireCsrMat.row_start);


    free(col_local_start_idx);
    free(col_local_length);
    free(val_local_start_idx);
    free(val_local_length);

}

void spmm_csr_v1(csrType_local csr_mat, denseType dense_mat, denseType *res_mat, \
                 double * global_swap_zone, int myid, int numprocs) 
{
    long ierr;
    long idx;
    // gather all data from all processes
    int recv_count[numprocs];
    int displs[numprocs];

    long local_num_row_normal = dense_mat.global_num_row / numprocs;
    long local_num_col_normal = dense_mat.global_num_col;
    long normal_num_elements = local_num_row_normal * local_num_col_normal;

    // values allocated by calloc() is initialized to zero
    double *res_buffer = (double *) calloc(res_mat->local_num_col * res_mat->local_num_row, \
                          sizeof (double));

    for (idx = 0; idx < numprocs; idx++) {
        recv_count[idx] = (int)normal_num_elements;
        displs[idx] = (int)(idx * normal_num_elements);

        if (idx == (numprocs - 1)) {
            recv_count[idx] = (int)((dense_mat.global_num_row - local_num_row_normal * (numprocs - 1))
                    * local_num_col_normal);
        }
    }
    //    ierr = MPI_Barrier(MPI_COMM_WORLD);
    // int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //                void *recvbuf, const int *recvcounts, const int *displs,
    //                MPI_Datatype recvtype, MPI_Comm comm)
    ierr = MPI_Allgatherv((void *) dense_mat.data, (int)dense_mat.local_num_col * dense_mat.local_num_row \
                       , MPI_DOUBLE, (void*)global_swap_zone, (int*)recv_count, (int*)displs \
                       , MPI_DOUBLE, MPI_COMM_WORLD);

    // spmm using csr format

    long idx_row;

    for (idx_row = 0; idx_row < csr_mat.num_rows; idx_row++) {
        long row_start_idx = csr_mat.row_start[idx_row];
        long row_end_idx = csr_mat.row_start[idx_row + 1];

        long idx_data;
        for (idx_data = row_start_idx; idx_data < row_end_idx; idx_data++) {
            long col_idx = csr_mat.col_idx[idx_data];
            double csr_data = csr_mat.csrdata[idx_data];

            long block_size = dense_mat.local_num_col;
            long block_idx;
            for (block_idx = 0; block_idx < block_size; block_idx++) {

                res_buffer[idx_row * res_mat->local_num_col + block_idx] +=
                        csr_data * global_swap_zone[col_idx * dense_mat.local_num_col + block_idx];
            }
        }
    }
    if (res_mat->data != 0) {
        free(res_mat->data);
    } else {
        exit(0);
    }
    res_mat->data = res_buffer;
}

// requirement: the changing elements in dataSrc should be allocated consecutively
// actually, this is a SpMV calculation
void spmm_csr_info_data_sep_CBCG(csrType_local csr_mat, denseType dense_mat_info, double * dataSrc, long dataDisp
        , denseType *res_mat, int myid, int numprocs) {

    int ierr;
    long idx;
    // gather all data from all processes
    int recv_count[numprocs];
    int displs[numprocs];

    long local_num_row_normal = dense_mat_info.global_num_row / numprocs;
    long local_num_col_normal = 1;
    long normal_num_elements = local_num_row_normal * local_num_col_normal;

    // recvBuf
    double * recvBuf = (double*)calloc( dense_mat_info.global_num_col * dense_mat_info.global_num_row, sizeof(double));
    // values allocated by calloc() is initialized to zero
    double *res_buffer = (double *) calloc(res_mat->local_num_col * res_mat->local_num_row, sizeof (double));

    for (idx = 0; idx < numprocs; idx++) {
        recv_count[idx] = (int)normal_num_elements;
        displs[idx] = (int)(idx * normal_num_elements);

        if (idx == (numprocs - 1)) {
            recv_count[idx] = (int)((dense_mat_info.global_num_row - local_num_row_normal * (numprocs - 1))
                    * local_num_col_normal);
        }
    }

    // int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //                void *recvbuf, const int *recvcounts, const int *displs,
    //                MPI_Datatype recvtype, MPI_Comm comm) 
    ierr = MPI_Allgatherv((void *) (dataSrc+dataDisp), (int)(dense_mat_info.local_num_col * dense_mat_info.local_num_row), MPI_DOUBLE \
            , (void *)recvBuf, (int *)recv_count, (int*)displs \
            , MPI_DOUBLE, MPI_COMM_WORLD);


    // spmv using csr format
    long idx_row;
    for (idx_row = 0; idx_row < csr_mat.num_rows; idx_row++) {
        long row_start_idx = csr_mat.row_start[idx_row];
        long row_end_idx = csr_mat.row_start[idx_row + 1];

        long idx_data;
        for (idx_data = row_start_idx; idx_data < row_end_idx; idx_data++) {
            long col_idx = csr_mat.col_idx[idx_data];
            double csr_data = csr_mat.csrdata[idx_data];

            res_buffer[idx_row] += csr_data * recvBuf[col_idx];
        }
    }
    if (res_mat->data != 0) {
        free(res_mat->data);
    } else {
        exit(0);
    }
    res_mat->data = res_buffer;

    free (recvBuf);

}

// change this version to monomial bases generation
// for monimail bases, s_alpha=1.0, s_beta=0.0
void mpk_v1(denseType S_mat, csrType_local mat, denseType R, long s, \
            double s_alpha, double s_beta, int myid, int numprocs) 
{
    long ierr;

    // Chebyshev polynomial chunk size
    long cpLocalChunkSize = R.local_num_col * R.local_num_row;
    long S_displ;

    double * AR_global_shape_buffer = (double*) calloc(R.global_num_col * R.global_num_row, sizeof (double));
    //S_mat_dummy which is a transposed S_mat
    // note that the data in S_mat_dummy is actually S_mat tranpose, 
    // so the global(local)_num and global(local)_num are wrong,
    // you need to exchange corresponding value of col and row to use them appropriately
    denseType S_mat_dummy;
    get_same_shape_denseType(S_mat, &S_mat_dummy);
    S_mat_dummy.local_num_col = S_mat.local_num_row;
    S_mat_dummy.local_num_row = S_mat.local_num_col;
    S_mat_dummy.global_num_col = S_mat.global_num_row;
    S_mat_dummy.global_num_row = S_mat.global_num_col;


    // if A is SPD, A*R and A*S(,j) should be of the same shape as R
    denseType spmvBuf;
    get_same_shape_denseType(R, &spmvBuf);

    // for monomial s_alpha=s_alpha_by2= 1.0, s_beta = s_beta_by2 = .0
    // double s_alpha_by2 = 2.0 * s_alpha;
    // double s_beta_by2 = 2.0 * s_beta;
    double s_alpha_by2 = s_alpha;
    double s_beta_by2 = s_beta;


    long sIdx;
    for (sIdx = 0; sIdx < s; sIdx++) {
        if (sIdx == 0) {
            //S(:,1)=r
            dense_entry_copy_disp(R, 0, S_mat_dummy, 0, R.local_num_col * R.local_num_row);
        } else if (sIdx == 1) {
            //S(:,2) = s_alpha * A * r + s_beta * r;
            spmm_csr_v1(mat, R, &spmvBuf, AR_global_shape_buffer, myid, numprocs);
            // res = alpha*mat1 + beta*mat2, TP, third place
            S_displ = cpLocalChunkSize;
            dense_mat_mat_add_TP_targetDisp(spmvBuf, R, S_mat_dummy, S_displ, s_alpha, s_beta, myid, numprocs);
        } else {
            //S(:,sIdx) = s_alpha_by2 * A * S(:,j-1) + s_beta_by2 * S(:,j-1) - S(:,j-2);
            S_displ = (sIdx - 1) * cpLocalChunkSize;
            spmm_csr_info_data_sep_CBCG(mat, R, S_mat_dummy.data, S_displ, &spmvBuf, myid, numprocs);
            dense_array_mat_mat_mat_add_TP_disp(spmvBuf.data, 0
                    , S_mat_dummy.data, (sIdx - 1) * cpLocalChunkSize
                    , S_mat_dummy.data, (sIdx - 2) * cpLocalChunkSize
                    , S_mat_dummy.data, cpLocalChunkSize * sIdx
                    , cpLocalChunkSize
                    , s_alpha_by2, s_beta_by2, -0.0
                    , myid, numprocs);
        }
    }
    // local transpose, and save to S_mat
    // so, matrix S_mat will be distributed matrix stored by row order
    dense_matrix_local_transpose_row_order(S_mat_dummy, S_mat);

    //
    free(S_mat_dummy.data);
    free(spmvBuf.data);
    free(AR_global_shape_buffer);

}

void spmv_csr_v3(csrType_local smat, denseType dmat, denseType resMat, \
                 long *sendIdxBuf, int *bufSendingCount, int *bufSendingDispls, \
                 double *sendVecDataBuf,int *remoteVecCount, int* remoteVecPtr,\
                 double *remoteVecDataBuf, int sendCount, int recvCount,int myid, int numprocs) 
{
    // double * shiftRemoteVecDataBuf = remoteVecDataBuf + smat.num_rows;
    // int idx, ierr;

    // // resMat->data may have already have some results calculated before, 
    // //     we need this buffer to restore result now and replace resMat->data later on
    // // if we don't use this buffer, we then need to reinitilize all items in resMat->data
    // //     before real spmv calculation
    // // double *resBuf = (double *) calloc(resMat->local_num_col * resMat->local_num_row \
    // //                                     , sizeof (double));

    // // gather data for sending
    // for (idx=0;idx<sendCount; idx++)
    // {
    //     sendVecDataBuf[idx] = dmat.data[ sendIdxBuf[idx] ];
    // }


    // // int MPI_Alltoallv(const void *sendbuf, const int *sendcounts,
    // //               const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
    // //               const int *recvcounts, const int *rdispls, MPI_Datatype recvtype,
    // //               MPI_Comm comm)

    //     // send vec data to each processes
    // ierr = MPI_Alltoallv((void*)sendVecDataBuf, (int*)bufSendingCount, (int*)bufSendingDispls, \
    //                     MPI_DOUBLE, (void*)shiftRemoteVecDataBuf, (int*)remoteVecCount, \
    //                     (int*) remoteVecPtr, MPI_DOUBLE, MPI_COMM_WORLD);

    // //spmv computational kernel
    // long rowIdx, vecColIdx;
    // long rowStartIdx, rowEndIdx;
    // double vecData, matData;
    // for (rowIdx = 0; rowIdx<smat.num_rows; rowIdx++)
    // {
    //     rowStartIdx = smat.row_start[rowIdx];
    //     rowEndIdx   = smat.row_start[rowIdx+1];
    //     resMat.data[rowIdx] = 0.0;

    //     for (idx = rowStartIdx; idx<rowEndIdx; idx++)
    //     {
    //         vecColIdx = smat.col_idx[idx];
    //         matData = smat.csrdata[idx];

    //         if (vecColIdx < smat.num_rows) //local buffer
    //         {
    //             vecData = dmat.data[vecColIdx];
    //         }
    //         else  // remote vec data buffer
    //         {
    //             vecData = remoteVecDataBuf[vecColIdx];
    //         }

    //         resMat.data[rowIdx] += matData * vecData;
    //     }

    // }

}

// Determine the remote vectors needed, expand local Vector array to receive these.  
// Re-index local Column Index array to point to the right Vector Data entry

int prepareRemoteVec_spmv(csrType_local mat, int myid, int numprocs, \
            int *remoteVecCount, int *remoteVecPtr, long *remoteVecIndex)
{
    // int ierr;
    // int numRemoteVec = 0;
    // int idx;
    // int aveVecSize; 
    // int tempRemoteVecCount[numprocs];

    // for (idx=0;idx<numprocs;idx++)
    // {
    //     tempRemoteVecCount[idx] = 0;
    //     remoteVecCount[idx] =0;
    // }

    // if (myid != numprocs-1)
    // {
    //     aveVecSize = (int)mat.num_rows;
    // }
    // else
    // {
    //     // for symmetric matrix
    //     aveVecSize = (int)(mat.num_cols/numprocs);
    // }
    
    // // printf ("myid: %d, aveVecSize : %ld \n", myid, aveVecSize);
    // int processPtr, finalProcessIdx = numprocs-1;
    // long vecStart = mat.start;
    // long vecEnd   = mat.start+ mat.num_rows;

    // for (idx=0;idx<mat.nnz;idx++)
    // {
    //     if (mat.col_idx[idx]<vecStart || mat.col_idx[idx]>=vecEnd)
    //     {
    //         if ((int)(mat.col_idx[idx]/aveVecSize) >= finalProcessIdx)
    //             processPtr = finalProcessIdx;
    //         else
    //             processPtr = (int)(mat.col_idx[idx]/aveVecSize);

    //         remoteVecCount[processPtr]++;
    //         numRemoteVec++;
    //     }

    // }

    // remoteVecPtr[0] = 0;    

    // for (idx = 1;idx<numprocs;idx++)
    // {
    //     remoteVecPtr[idx] = remoteVecPtr[idx-1]+remoteVecCount[idx-1];
    // }

    // for (idx=0;idx<mat.nnz;idx++) 
    // {
    //     // remotely
    //     if (mat.col_idx[idx]<vecStart || mat.col_idx[idx]>=vecEnd)
    //     {
    //         if ((int)(mat.col_idx[idx]/aveVecSize) >= finalProcessIdx)
    //             processPtr = finalProcessIdx;
    //         else
    //             processPtr = (int)(mat.col_idx[idx]/aveVecSize);

    //         remoteVecIndex[remoteVecPtr[processPtr] + tempRemoteVecCount[processPtr]] = 
    //                         mat.col_idx[idx] - processPtr*aveVecSize;

    //         mat.col_idx[idx] = mat.num_rows + remoteVecPtr[processPtr] + tempRemoteVecCount[processPtr];
    //         // mat.col_idx[idx] = remoteVecPtr[processPtr] + tempRemoteVecCount[processPtr];
    //         tempRemoteVecCount[processPtr] ++;

    //     }
    //     else //locally
    //     {
    //         mat.col_idx[idx] -= vecStart;
    //     }

    // }
    
    // return numRemoteVec;

}

// refer to mpk_v2, change this version to monomial bases generation
void mpk_v2(csrType_local mat, denseType B, denseType X, long s, double epsilon, \
    int myid, int numprocs) 
{

    // int db_idx;
    // int generalIdx;
    // int ierr;

    // long *remoteVecIndex = (long*)calloc(mat.nnz, sizeof(long));

    // // if something happened, check if here are all initiallized to zeros
    // int remoteVecCount[numprocs], remoteVecPtr[numprocs];

    // int numRemoteVec =  prepareRemoteVec_spmv(mat, myid, numprocs,\
    //                      remoteVecCount, remoteVecPtr, remoteVecIndex);

    // // use remoteVecDataBuf for receive data from remote prcesses
    // double * remoteVecDataBuf = (double*) malloc ( (numRemoteVec+mat.num_rows)*sizeof(double));
    
    // int bufSendingCount[numprocs], bufSendingDispls[numprocs];


    // // int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    // //              void *recvbuf, int recvcount, MPI_Datatype recvtype,
    // //              MPI_Comm comm)
    // // this MPI call seems can be saved when the matrix is symmetric
    // // , and in that case, toSendTotal==numRemoteVec
    // ierr = MPI_Alltoall((void*)remoteVecCount, 1, MPI_INT, \
    //                     (void*)bufSendingCount,1, MPI_INT, MPI_COMM_WORLD);

    // int toSendTotal = 0;
    // for (generalIdx=0;generalIdx<numprocs;generalIdx++)
    // {
    //     toSendTotal+=bufSendingCount[generalIdx];
    // }

    
    // // buffer for vector data being sent
    // double * sendVecDataBuf   = (double*) malloc ( toSendTotal * sizeof(double));
    // long *sendIdxBuf  = (long*)malloc( toSendTotal * sizeof(long) );

    // bufSendingDispls[0] = 0;
    // for (generalIdx=1;generalIdx<numprocs;generalIdx++)
    // {
    //     bufSendingDispls[generalIdx] = bufSendingDispls[generalIdx-1] + \
    //                                    bufSendingCount[generalIdx-1];
    // }

    // // int MPI_Alltoallv(const void *sendbuf, const int *sendcounts,
    // //               const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
    // //               const int *recvcounts, const int *rdispls, MPI_Datatype recvtype,
    // //               MPI_Comm comm)
    // ierr = MPI_Alltoallv( (void*)remoteVecIndex, (int*)remoteVecCount, (int*)remoteVecPtr,\
    //                        MPI_LONG, \
    //                        (void*)sendIdxBuf, (int*)bufSendingCount, (int*)bufSendingDispls,\
    //                        MPI_LONG, MPI_COMM_WORLD);

    // //A*X, Spmm when m=1
    // // spmm_csr_v1(mat, X, &A_X, B_global_shape_swap_zone, myid, numprocs);
    // spmv_csr_v3(mat, X, A_X, sendIdxBuf, bufSendingCount, bufSendingDispls, \
    //         sendVecDataBuf, remoteVecCount, remoteVecPtr, \
    //         remoteVecDataBuf,toSendTotal ,numRemoteVec ,myid, numprocs);


    // if (remoteVecDataBuf !=NULL && sendVecDataBuf !=NULL \
    //     && sendIdxBuf    !=NULL && remoteVecIndex !=NULL)
    // {
    //     free (remoteVecIndex);
    //     free(remoteVecDataBuf);
    //     free (sendVecDataBuf);
    //     free (sendIdxBuf);

    // }
    // else
    // {
    //     printf("some of buffers are NULL\n");
    //     exit(1);
    // }


}

