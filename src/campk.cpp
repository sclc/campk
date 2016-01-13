//
#include "campk.h"


void campk_v1(denseType X, denseType &AkX , int kval 
                 , matInfo * mat_info, int myid, int numprocs 
                 , char* path, char* mtx_filename)
{
    cooType globalMat;
    csrType_local entireCsrMat;

    long idx;
    int ierr;
    int numRemoteVec;

    long infoMat[3];
    long myRowStart, myRowEnd, averageNumRowPerProc, myNumRow;
    // long involvedRowCounter = 0;
    // short *localRowFlag;

    std::map<long, std::vector<int> > dependencyRecoder;
    csrType_local_var compactedCSR;

    double *buffer_vec_remote_recv;
    long *vec_remote_recv_idx;

#ifdef CAMPK_PROF_ALL
    double t1, t2, t_past_local, t_past_global;
#endif

    if (myid == 0) 
    {
        // rank 0 read mtr

        readMtx_info_and_coo(path, mtx_filename, mat_info, & globalMat);

        printf("there are %ld elements in matrix \n", mat_info->nnz);

        Converter_Coo2Csr(globalMat, &entireCsrMat, mat_info);

        infoMat[0] = mat_info->num_rows;
        infoMat[1] = mat_info->num_cols;
        infoMat[2] = mat_info->nnz;

        delete_cooType(globalMat);

    }
    
    // so far entireCsrMat on rank0 has all matrix element 
    //, and store them in CSR format
    

    ierr = MPI_Bcast((void *) infoMat, 3, MPI_LONG, 0, MPI_COMM_WORLD);

    if (myid != 0)
    {
        entireCsrMat.num_rows = infoMat[0];
        entireCsrMat.num_cols = infoMat[1];
        entireCsrMat.nnz      = infoMat[2];

        entireCsrMat.row_start = (long*)malloc ( (entireCsrMat.num_rows+1) * sizeof(long));
        entireCsrMat.col_idx   = (long*)malloc (  entireCsrMat.nnz * sizeof(long));
        entireCsrMat.csrdata   = (double *)malloc (entireCsrMat.nnz * sizeof(double));

        assert (entireCsrMat.row_start != NULL);
        assert (entireCsrMat.col_idx   != NULL); 
        assert (entireCsrMat.csrdata   != NULL);

    }

    // boardcast rowIdx
    ierr = MPI_Bcast((void *) entireCsrMat.row_start, (entireCsrMat.num_rows+1), 
                      MPI_LONG, 0, MPI_COMM_WORLD);
    // boardcast colIdx
    ierr = MPI_Bcast((void *) entireCsrMat.col_idx, entireCsrMat.nnz, 
                      MPI_LONG, 0, MPI_COMM_WORLD);
    // boardcast data
    ierr = MPI_Bcast((void *) entireCsrMat.csrdata, entireCsrMat.nnz, 
                      MPI_LONG, 0, MPI_COMM_WORLD);    

    averageNumRowPerProc = (long)(entireCsrMat.num_rows / numprocs);
    myNumRow = averageNumRowPerProc;

    myRowStart = myid * averageNumRowPerProc;
    myRowEnd   = myRowStart+myNumRow - 1;

    if (myid == numprocs - 1)
    {
        myNumRow = entireCsrMat.num_rows - myid * averageNumRowPerProc;
        myRowEnd = entireCsrMat.num_rows - 1;
    }
    
    AkX.local_num_row = myNumRow * kval;
    AkX.local_num_col = 1;
    AkX.global_num_row = infoMat[0] * kval;
    AkX.global_num_col = 1;
    AkX.start_idx = myRowStart;

    // compute locally computable elements
    long vec_result_length = AkX.local_num_row * AkX.local_num_col;
    double *k_level_result = (double*) calloc ( vec_result_length,sizeof(double) );

    // this should be replaced by real x values *****
    assert (X.local_num_col == 1);
    assert (X.local_num_row == myNumRow);
    for (idx = 0; idx<myNumRow; idx++)
    {
        // k_level_result[idx] = (double)(myRowStart + idx+10);
        k_level_result[idx] = X.data[idx];
    }

#ifdef DB_CAMPK_V1_1
    printf ("myid: %d. in campk_v1 matrix dist done\n", myid);
    ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef CAMPK_PROF_DEP_BFS
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
#endif

/////////////////////////////////
    short  *k_level_locally_computable_flags = (short *) calloc (myNumRow * kval, sizeof(short) );

    // campk_local_dependecy_BFS_v1 (entireCsrMat, dependencyRecoder, myNumRow, myRowStart, myRowEnd, k_level_locally_computable_flags, kval\
    //                             , myid, numprocs);
    campk_local_dependecy_BFS_v2 (entireCsrMat, dependencyRecoder, myNumRow, myRowStart, myRowEnd, k_level_locally_computable_flags, kval\
                                , myid, numprocs);
////////////////////////////////
#ifdef CAMPK_PROF_DEP_BFS
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    t_past_local = t2 - t1;
    ierr = MPI_Reduce(&t_past_local, &t_past_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myid == 0)
    {
        // printf ("campk_local_dependecy_BFS_v1 overhead: %lf sec\n", t_past_global);
        printf ("campk_local_dependecy_BFS_v2 overhead: %lf sec\n", t_past_global);
    }
#endif

#ifdef DB_CAMPK_V1_1
    printf ("myid: %d. in campk_v1 campk_local_dependecy_BFS_v1 done\n", myid);
    ierr = MPI_Barrier(MPI_COMM_WORLD);

#endif

#ifdef CAMPK_PROF_COMPACT
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
#endif

//////////////////////////////// actual code
    campk_compacting_csr_v1 (compactedCSR, dependencyRecoder, entireCsrMat, myid, numprocs);
////////////////////////////////

#ifdef CAMPK_PROF_COMPACT
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    t_past_local = t2 - t1;
    ierr = MPI_Reduce(&t_past_local, &t_past_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myid == 0)
    {
        printf ("campk_compacting_csr_v1 overhead: %lf sec\n", t_past_global);
    }
#endif

#ifdef DB_CAMPK_V1_1
    printf ("myid: %d. in campk_v1 campk_compacting_csr_v1 done\n", myid);
    ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef CAMPK_PROF_COMM_COMPUT
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
#endif
////////////////////////////////////// actual code start
    numRemoteVec = campk_comm_overlaping_local_computation_v1 (compactedCSR, dependencyRecoder, buffer_vec_remote_recv 
                                                             , k_level_locally_computable_flags, k_level_result, vec_result_length 
                                                             , myNumRow, myRowStart, myRowEnd, averageNumRowPerProc, vec_remote_recv_idx 
                                                             , myid, numprocs);
///////////////////////////////////// actual code done
#ifdef CAMPK_PROF_COMM_COMPUT
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    t_past_local = t2 - t1;
    ierr = MPI_Reduce(&t_past_local, &t_past_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myid == 0)
    {
        printf ("campk_comm_overlaping_local_computation_v1 overhead: %lf sec\n", t_past_global);
    }
#endif

#ifdef DB_CAMPK_V1_1
    printf ("myid: %d. in campk_v1 campk_comm_overlaping_local_computation_v1 done. numRemoteVec:%d\n", myid,numRemoteVec);
    ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif

// ierr = MPI_Barrier(MPI_COMM_WORLD);
// exit(1);
#ifdef CAMPK_PROF_AFTER_COMM_COMPUT
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
#endif
////////////////////////////////////// actual code start
    // wait for communicaiton done
    // when communication finished, computate locally incomputable items
  //  campk_after_comm_computation_v1 (compactedCSR, k_level_result, k_level_locally_computable_flags 
  //                              , vec_result_length, myNumRow, myRowStart, myRowEnd, vec_remote_recv_idx, buffer_vec_remote_recv 
  //                              , numRemoteVec, myid, numprocs);

    campk_after_comm_computation_v2 (compactedCSR, k_level_result, k_level_locally_computable_flags 
                                , vec_result_length, myNumRow, myRowStart, myRowEnd, vec_remote_recv_idx, buffer_vec_remote_recv 
                                , numRemoteVec, kval,myid, numprocs);
///////////////////////////////////// actual code done
#ifdef CAMPK_PROF_AFTER_COMM_COMPUT
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    t_past_local = t2 - t1;
    ierr = MPI_Reduce(&t_past_local, &t_past_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myid == 0)
    {
        printf ("campk_after_comm_computation overhead: %lf sec\n", t_past_global);
    }
#endif

#ifdef DB_CAMPK_V1_1
    printf ("myid: %d. in campk_v1 campk_after_comm_computation done\n", myid);
    ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif
    // set up result
    AkX.data = k_level_result;

 // free memory
    // free (k_level_result);
    free (k_level_locally_computable_flags);
    free (buffer_vec_remote_recv);
    free (vec_remote_recv_idx);
}

//
// compute data in dependencyRecoder and k_level_locally_computable_flags
// #define MEM_USAGE_1
// #define MEM_USAGE_2
void campk_local_dependecy_BFS_v1 (csrType_local entireCsrMat, std::map<long, std::vector<int> > &dependencyRecoder 
                                 , long myNumRow, long myRowStart, long myRowEnd, short * k_level_locally_computable_flags, int kval 
                                 , int myid, int numprocs)
{
    // 0: irrevelent; 1: revelent; 2: last level
    // , all initialize to 0
    // localRowFlag = (short *)calloc (entireCsrMat.num_rows, sizeof(short));
    long localRowIdx;
    std::queue<long> rowIdxScanQueue;
    // std::map<long, std::vector<int> > dependencyRecoder;
    long eleStartIdx, eleEndIdx, eleIdx;
    long levelNumEOld=0, levelNumENew=0;
    long queueRowIdxTemp;
    int idx;
    int ierr;

    //
    for (localRowIdx=myRowStart; localRowIdx <= myRowEnd; localRowIdx++)
    {
        rowIdxScanQueue.push(localRowIdx);
        levelNumEOld++;
    }

    assert (levelNumEOld == rowIdxScanQueue.size());

#ifdef MEM_USAGE_1
    int mapSize = 0, queueSize=0, idxchecker=10000;
#endif
    for (idx = kval; idx>0; idx--)
    {
        // level idx being out of queue
        //, level idx+1 go into queue
        //, BFS search
        while (levelNumEOld-- !=0 )
        {
            queueRowIdxTemp = rowIdxScanQueue.front();
            rowIdxScanQueue.pop();

            eleStartIdx = entireCsrMat.row_start[queueRowIdxTemp];
            eleEndIdx   = entireCsrMat.row_start[queueRowIdxTemp+1];

            for (eleIdx = eleStartIdx; eleIdx<eleEndIdx; eleIdx++)
            {
                // A^(idx-1)X's element are needed to compute A^(idx)X
                dependencyRecoder[entireCsrMat.col_idx[eleIdx]].push_back(idx-1);
                // push in row index for idx level BFS
                rowIdxScanQueue.push(entireCsrMat.col_idx[eleIdx]);
                levelNumENew++;
#ifdef MEM_USAGE_1
    if (mapSize*2 < dependencyRecoder.size() || queueSize*2 < rowIdxScanQueue.size() || idxchecker!=idx)
    {
        int vecIdx, vecSizeMax=0;
        mapSize = dependencyRecoder.size();
        queueSize = rowIdxScanQueue.size();
        idxchecker =idx;
        printf ("myid:%d, idx:%d ,dependencyRecoder.size:%d ,rowIdxScanQueue.size:%d\n", myid, idx,dependencyRecoder.size(), rowIdxScanQueue.size() );

    }
    
    // int ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif
            }
        }
        // make sure all old elements index are out of queue
        assert (levelNumENew == rowIdxScanQueue.size());

#ifdef MEM_USAGE_1
        printf ("myid: %d, levelidx: %d, levelNumENew:%d, rowIdxScanQueue.size:%d\n", myid, idx,levelNumENew, rowIdxScanQueue.size());
        ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif
        levelNumEOld = levelNumENew;
        levelNumENew = 0; 
    }
#ifdef MEM_USAGE_2
    // ClearQueue (rowIdxScanQueue);
    printf ("myid: %d, going to analyze locally computability \n", myid);

    ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif

    long k_level_result_lda = myNumRow;
    long result_ptr_this, result_ptr_last, result_ptr_last_offset;
    short computable_flag = 1;

    for (idx = 1; idx<kval; idx++)
    {
        result_ptr_this = idx * k_level_result_lda;
        result_ptr_last = (idx - 1) * k_level_result_lda;

        for (localRowIdx=myRowStart; localRowIdx <= myRowEnd; localRowIdx++)
        {
            computable_flag = 1;
            eleStartIdx = entireCsrMat.row_start[localRowIdx];
            eleEndIdx   = entireCsrMat.row_start[localRowIdx+1];


            for (eleIdx = eleStartIdx; eleIdx<eleEndIdx; eleIdx++)
            { 
                result_ptr_last_offset = (idx - 1) * k_level_result_lda + \
                                        entireCsrMat.col_idx[eleIdx] - myRowStart;
                // The first condition can be removed for matrix that has no zero diagonal items
                // , 
                if (k_level_locally_computable_flags[result_ptr_last] < (idx -1) || \
                    entireCsrMat.col_idx[eleIdx] < myRowStart || 
                    entireCsrMat.col_idx[eleIdx] >  myRowEnd)
                {
                    computable_flag = 0;
                    // continue;
                    break;
                }
                // you have give a condition branch here
                //,  cause sometimes result_ptr_last_offset < 0
                //,  segment falut will happen, you have rule out this condtion in 
                if (k_level_locally_computable_flags[result_ptr_last_offset] < (idx -1) )
                {
                    computable_flag = 0;
                    // continue;
                    break;
                }

            }
            // locally computable
            if (computable_flag)
            {
                    k_level_locally_computable_flags[result_ptr_this] = 
                    k_level_locally_computable_flags[result_ptr_last]+1;
            }
            else
            {
                k_level_locally_computable_flags[result_ptr_this] = 
                k_level_locally_computable_flags[result_ptr_last] ;
            }

            result_ptr_this++;
            result_ptr_last++;
        }
    }
#ifdef MEM_USAGE_2
    // ierr = MPI_Barrier(MPI_COMM_WORLD);
    // exit(1);
#endif
}

////
void campk_local_dependecy_BFS_v2 (csrType_local entireCsrMat, std::map<long, std::vector<int> > &dependencyRecoder 
                                 , long myNumRow, long myRowStart, long myRowEnd, short * k_level_locally_computable_flags, int kval 
                                 , int myid, int numprocs)
{
    // 0: irrevelent; 1: revelent; 2: last level
    // , all initialize to 0
    // localRowFlag = (short *)calloc (entireCsrMat.num_rows, sizeof(short));
    long localRowIdx;
    // std::queue<long> rowIdxScanQueue;
    long *reachableRecorderOld, *reachableRecorderNew, *reachableRecorderTemp;
    long recorderLength = entireCsrMat.num_rows, recorderIdx;
    reachableRecorderOld = (long*)calloc( recorderLength, sizeof(long) );
    reachableRecorderNew = (long*)calloc( recorderLength, sizeof(long) );
    // std::map<long, std::vector<int> > dependencyRecoder;
    long eleStartIdx, eleEndIdx, eleIdx;
    // long levelNumEOld=0, levelNumENew=0;
    long queueRowIdxTemp;
    int idx;
    int ierr;

    //
    for (localRowIdx=myRowStart; localRowIdx <= myRowEnd; localRowIdx++)
    {
        reachableRecorderOld[localRowIdx] +=1;
        // levelNumEOld++;
    }

#ifdef MEM_USAGE_1
    int mapSize = 0, queueSize=0, idxchecker=10000;
#endif
    for (idx = kval; idx>0; idx--)
    {
        // level idx being out of queue
        //, level idx+1 go into queue
        //, BFS search
        for (recorderIdx=0; recorderIdx< recorderLength; recorderIdx++)
        {

            if (reachableRecorderOld[recorderIdx] == 0) continue;

            // levelNumEOld--;

            queueRowIdxTemp = recorderIdx;

            eleStartIdx = entireCsrMat.row_start[queueRowIdxTemp];
            eleEndIdx   = entireCsrMat.row_start[queueRowIdxTemp+1];

            for (eleIdx = eleStartIdx; eleIdx<eleEndIdx; eleIdx++)
            {
                // A^(idx-1)X's element are needed to compute A^(idx)X
                dependencyRecoder[entireCsrMat.col_idx[eleIdx]].push_back(idx-1);
                // push in row index for idx level BFS
                reachableRecorderNew[ entireCsrMat.col_idx[eleIdx] ] +=1;

#ifdef MEM_USAGE_1
    if (mapSize*2 < dependencyRecoder.size()|| idxchecker!=idx)
    {
        int vecIdx, vecSizeMax=0;
        mapSize = dependencyRecoder.size();
        idxchecker =idx;
        printf ("myid:%d, idx:%d ,dependencyRecoder.size:%d \n", myid, idx, dependencyRecoder.size() );
    }

#endif
            }
        }

        // this level is over, reset memory
        reachableRecorderTemp = reachableRecorderOld;
        reachableRecorderOld = reachableRecorderNew;
        reachableRecorderNew = reachableRecorderTemp;        
        memset(reachableRecorderNew, 0, recorderLength);
    }
    free (reachableRecorderNew);
    free (reachableRecorderOld);

#ifdef MEM_USAGE_2
    // ClearQueue (rowIdxScanQueue);
    printf ("myid: %d, going to analyze locally computability \n", myid);

    ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif

    long k_level_result_lda = myNumRow;
    long result_ptr_this, result_ptr_last, result_ptr_last_offset;
    short computable_flag = 1;

    for (idx = 1; idx<kval; idx++)
    {
        result_ptr_this = idx * k_level_result_lda;
        result_ptr_last = (idx - 1) * k_level_result_lda;

        for (localRowIdx=myRowStart; localRowIdx <= myRowEnd; localRowIdx++)
        {
            computable_flag = 1;
            eleStartIdx = entireCsrMat.row_start[localRowIdx];
            eleEndIdx   = entireCsrMat.row_start[localRowIdx+1];


            for (eleIdx = eleStartIdx; eleIdx<eleEndIdx; eleIdx++)
            { 
                result_ptr_last_offset = (idx - 1) * k_level_result_lda + 
                                        entireCsrMat.col_idx[eleIdx] - myRowStart;

                if (k_level_locally_computable_flags[result_ptr_last] < (idx -1) || 
                    entireCsrMat.col_idx[eleIdx] < myRowStart || 
                    entireCsrMat.col_idx[eleIdx] >  myRowEnd)
                {
                    computable_flag = 0;
                    break;
                    // continue;
                }
                // you have give a condition branch here
                //,  cause sometimes result_ptr_last_offset < 0
                //,  segment falut will happen, you have rule out this condtion in 
                if (k_level_locally_computable_flags[result_ptr_last_offset] < (idx -1) )
                {
                    computable_flag = 0;
                    break;
                    // continue;
                }

            }
            // locally computable
            if (computable_flag)
            {
                    k_level_locally_computable_flags[result_ptr_this] = 
                    k_level_locally_computable_flags[result_ptr_last]+1;
            }
            else
            {
                k_level_locally_computable_flags[result_ptr_this] = 
                k_level_locally_computable_flags[result_ptr_last] ;
            }

            result_ptr_this++;
            result_ptr_last++;
        }
    }
#ifdef MEM_USAGE_2
    // ierr = MPI_Barrier(MPI_COMM_WORLD);
    // exit(1);
#endif
}

//
// compute data in compactedCSR
void campk_compacting_csr_v1 (csrType_local_var &compactedCSR, std::map<long, std::vector<int> > dependencyRecoder 
                           , csrType_local entireCsrMat, int myid, int numprocs)
{
    // compacting local CSR
    // csrType_local_var compactedCSR;
    compactedCSR.num_rows = dependencyRecoder.size();
    compactedCSR.num_cols = entireCsrMat.num_cols;

    std::map<long, std::vector<int> > ::iterator mapIterIdx;
    long rowIdxTemp;
    long eleStartIdx, eleEndIdx, eleIdx;

    for (mapIterIdx = dependencyRecoder.begin(); mapIterIdx!=dependencyRecoder.end(); 
         ++mapIterIdx)
    {
        // assume k > 1, here
        //, so we at least will compute A^2x, Ax
        // if some rows only related to A^0x, these rows is not necessarily to be stored
        if (mapIterIdx->second[0] == 0) continue;

        rowIdxTemp = mapIterIdx->first;
        compactedCSR.row_start[rowIdxTemp] = compactedCSR.col_idx.size();

        eleStartIdx = entireCsrMat.row_start[rowIdxTemp];
        eleEndIdx   = entireCsrMat.row_start[rowIdxTemp+1];

        for (eleIdx = eleStartIdx; eleIdx<eleEndIdx; eleIdx++)
        {
            compactedCSR.col_idx.push_back(entireCsrMat.col_idx[eleIdx]);
            compactedCSR.csrdata.push_back(entireCsrMat.csrdata[eleIdx]);
        }
        compactedCSR.row_end[rowIdxTemp] = compactedCSR.col_idx.size() - 1;
    }

    assert ( compactedCSR.col_idx.size() == compactedCSR.csrdata.size() );

    compactedCSR.nnz = compactedCSR.col_idx.size();

    // free entireCsrMat space after constructed compactedCSR
    if (entireCsrMat.row_start != NULL) free (entireCsrMat.row_start);
    if (entireCsrMat.col_idx != NULL)   free (entireCsrMat.col_idx);
    if (entireCsrMat.csrdata != NULL)   free (entireCsrMat.csrdata);

}

// set up values in  buffer_vec_remote_recv and parts of k_level_result
// set up values in  vec_remote_recv_idx, numRemoteVec
// #define DB_COMM_LOCAL_COMP_V1

int campk_comm_overlaping_local_computation_v1 (csrType_local_var compactedCSR, std::map<long, std::vector<int> > dependencyRecoder 
                                               , double *& buffer_vec_remote_recv, short  *k_level_locally_computable_flags 
                                               , double *k_level_result, long vec_result_length, long myNumRow, long myRowStart, long myRowEnd 
                                               , long averageNumRowPerProc, long *& vec_remote_recv_idx, int myid, int numprocs)
{
    //all dependencyRecoder.first are necessary elements of x to compute A^kx, .., A^1x
    //, so prepare buffers for MPI call to transfer this x elements
    // count parameters for MPI_Alltoallv have to be in int type
    int numRemoteVec=0, numSendingVec=0; // for preparing buffer to receive remote vec items
    int ptrRemoteVec[numprocs], remoteVecCount[numprocs];
    int sendingCount[numprocs];
    int sendingBufferPtr[numprocs];
    int tempRemoteVecCount[numprocs];

    int my_level;
    int num_level_computable;
    long resPtr;
    long myRowIdx, locally_computable_col_idx;
    double last_result_val;

    long idx;
    int ierr;
    long eleStartIdx, eleEndIdx;

    for (idx=0; idx<numprocs;idx++)
    {   
        tempRemoteVecCount[idx] = 0;
        remoteVecCount[idx] = 0;
    }


    std::map<long, std::vector<int> > ::iterator mapIterIdx_remote;
    long rowIdxRemoteChecking;
    int procPtr;
    // double *buffer_vec_remote_recv = NULL;
    double *buffer_vec_remote_sending = NULL;
    long *vec_remote_sending_idx = NULL;



// calculate remoteVecCount for processes and numRemoteVec for creating buffers
    for (mapIterIdx_remote = dependencyRecoder.begin(); mapIterIdx_remote!=dependencyRecoder.end(); ++mapIterIdx_remote)
    {
        rowIdxRemoteChecking = mapIterIdx_remote->first;
        if ( rowIdxRemoteChecking < myRowStart || rowIdxRemoteChecking > myRowEnd)
        {
            procPtr = (int) rowIdxRemoteChecking/averageNumRowPerProc;
            procPtr = procPtr > (numprocs - 1) ? (numprocs - 1): procPtr;
            remoteVecCount[ procPtr ] ++;
            numRemoteVec++;
        }

    }
#ifdef DB_COMM_LOCAL_COMP_V1
    printf ("myid: %d.  communication done.numRemoteVec:%d\n", myid,numRemoteVec);
    ierr = MPI_Barrier(MPI_COMM_WORLD);


    printf ("function mission accomplished\n");
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    exit(1);
#endif
    buffer_vec_remote_recv = (double *)malloc (numRemoteVec * sizeof(double));
    vec_remote_recv_idx  =     (long *)malloc (numRemoteVec * sizeof(long));

    ptrRemoteVec[0] = 0;
    for (idx = 1; idx<numprocs;idx++ )
    {
        ptrRemoteVec[idx] = ptrRemoteVec[idx-1] + remoteVecCount[idx-1];
    }

// restore idx data to buffer for communicating
    for (mapIterIdx_remote = dependencyRecoder.begin(); mapIterIdx_remote!=dependencyRecoder.end(); ++mapIterIdx_remote)
    {
        rowIdxRemoteChecking = mapIterIdx_remote->first;
        if ( rowIdxRemoteChecking < myRowStart || rowIdxRemoteChecking > myRowEnd)
        {
            procPtr = (int) rowIdxRemoteChecking/averageNumRowPerProc;
            procPtr = procPtr > (numprocs - 1) ? (numprocs - 1): procPtr;

            // some redundant, may use map<key,val> to remove this 
            //, optimize this part later on
            vec_remote_recv_idx[ptrRemoteVec[procPtr] + tempRemoteVecCount[procPtr] ] = 
                                rowIdxRemoteChecking;
            tempRemoteVecCount[procPtr]++;
        }

    }

    // int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
    //                  MPI_Comm comm)

    ierr = MPI_Alltoall( (void *)remoteVecCount, 1, MPI_INT,
                         (void *)sendingCount, 1, MPI_INT, MPI_COMM_WORLD);


    sendingBufferPtr[0] = 0;
    numSendingVec += sendingCount[0];
    for (idx =1; idx<numprocs; idx++)
    {
        sendingBufferPtr[idx] = sendingBufferPtr[idx-1] + sendingCount[idx-1];
        numSendingVec += sendingCount[idx];
    }
    

    buffer_vec_remote_sending = (double *) malloc (numSendingVec*sizeof(double));
    vec_remote_sending_idx =       (long *)malloc (numSendingVec*sizeof(double));

    // int MPI_Alltoallv(const void *sendbuf, const int *sendcounts,
    //               const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
    //               const int *recvcounts, const int *rdispls, MPI_Datatype recvtype,
    //               MPI_Comm comm)

    // to tell each process which idx I need
    // , in bcbcg solver, this will only be applied once
    // , so it does not matter wether it is syn or asyn
    ierr =  MPI_Alltoallv((void *) vec_remote_recv_idx, (int*) remoteVecCount, 
                            (int*) ptrRemoteVec, MPI_LONG, 
                          (void *) vec_remote_sending_idx, (int*) sendingCount, 
                            (int*) sendingBufferPtr,  MPI_LONG, 
                            MPI_COMM_WORLD);

    // to prepare buffer_vec_remote_sending for sending
    long k_level_result_idx;
    for (idx =0; idx< numSendingVec; idx++)
    {
        k_level_result_idx = vec_remote_sending_idx[idx];
        assert (k_level_result_idx >= myRowStart && k_level_result_idx <= myRowEnd);

        buffer_vec_remote_sending[idx] = k_level_result[k_level_result_idx-myRowStart];
    }
    //to tell each processes the vals they need
    // , in bcbcg solver, this will be applied in each spmv call
    // , so it should be asyn 
    // , to optimize this part later on
    //  if I want to use asynchronous send receive here
    // , then I can not use MPI_Alltoallv
    //I need to build a communicaiton table firstly
    // , then do asynchronous p2p send recv
    //based on matrix sparsity patterns, a dozens of send-recev may be called
    // , and I am wondering how the performance will be in such cases
    ierr =  MPI_Alltoallv((void *) buffer_vec_remote_sending, (int*) sendingCount, 
                            (int*) sendingBufferPtr, MPI_DOUBLE, 
                          (void *) buffer_vec_remote_recv, (int*) remoteVecCount, 
                            (int*) ptrRemoteVec, MPI_DOUBLE, 
                            MPI_COMM_WORLD);


    //locally computatble
        // compute locally computable
    // ...

    // in my campk design
    //, the first myNumRow items in k_level_result are actually the original x in MPK
    for (resPtr=myNumRow; resPtr<vec_result_length; resPtr++)
    {
        num_level_computable =  (int)k_level_locally_computable_flags[resPtr];
        my_level = (int) (resPtr / myNumRow);

        if (num_level_computable == my_level)
        {
            myRowIdx = (resPtr%myNumRow) + myRowStart;
            eleStartIdx = compactedCSR.row_start[myRowIdx];
            eleEndIdx   = compactedCSR.row_end[myRowIdx];

            for (idx = eleStartIdx; idx<= eleEndIdx; idx++)
            {
                locally_computable_col_idx = compactedCSR.col_idx[idx];
                last_result_val = k_level_result[ (my_level - 1)* myNumRow + 
                                                    locally_computable_col_idx - 
                                                    myRowStart];
                k_level_result[resPtr] += compactedCSR.csrdata[idx] * last_result_val;
            }

        }
        else 
        {
            continue;
        }

    }

    // free buffers
    free (buffer_vec_remote_sending);
    free (vec_remote_sending_idx);

    return numRemoteVec;
}

///////////////////////////////////
void campk_after_comm_computation_v1 (csrType_local_var compactedCSR, double *k_level_result, short  *k_level_locally_computable_flags 
                                 , long vec_result_length, long myNumRow, long myRowStart, long myRowEnd,long *vec_remote_recv_idx 
                                 , double * buffer_vec_remote_recv, int numRemoteVec, int myid, int numprocs)
{

#ifdef CAMPK_PROF_AFTER_COMM_COMPUT_FUNC_ALL
    double t1, t2, t_past_local=0.0, t_past_localtotal=0.0;
    int ierr;
#endif

    long locally_incomputable_col_idx;
    long remoteEleStartIdx, remoteEleEndIdx;
    long remoteEleIdx;
    long remoteLocallyComputableColIdx;
    std::map<long, std::vector<double> > remoteValResultRecoder;

    long idx;
    long eleStartIdx, eleEndIdx;
    long resPtr, myRowIdx;

    int my_level;
    int num_level_computable;

    double last_result_val;

    bool recursiveChecker=false;

    for (idx=0; idx<numRemoteVec;idx++)
    {
        if (remoteValResultRecoder.find (vec_remote_recv_idx[idx]) == remoteValResultRecoder.end())
        {
            remoteValResultRecoder[vec_remote_recv_idx[idx]].push_back(buffer_vec_remote_recv[idx]);
        }
        assert (remoteValResultRecoder[vec_remote_recv_idx[idx]].size() == 1);
    }



    // my_level > 0
    for (resPtr=myNumRow; resPtr<vec_result_length; resPtr++)
    {
        num_level_computable =  (int)k_level_locally_computable_flags[resPtr];

        my_level = (int) (resPtr / myNumRow);

        if (num_level_computable != my_level)
        {
            myRowIdx = (resPtr%myNumRow) + myRowStart;
            eleStartIdx = compactedCSR.row_start[myRowIdx];
            eleEndIdx   = compactedCSR.row_end[myRowIdx];

            for (idx = eleStartIdx; idx<= eleEndIdx; idx++)
            {
                locally_incomputable_col_idx = compactedCSR.col_idx[idx];
#ifdef CAMPK_PROF_AFTER_COMM_COMPUT_FUNC_1
    t1 = MPI_Wtime();
#endif
//////////////////////////////////// actural code starts
                recursiveChecker = RecursiveDependentEleComputation(remoteValResultRecoder, compactedCSR, 
                            locally_incomputable_col_idx, my_level, k_level_result, myRowStart, myRowEnd, myNumRow,
                            myid, numprocs );
//////////////////////////////////// actural code ends
#ifdef CAMPK_PROF_AFTER_COMM_COMPUT_FUNC_1
    t2 = MPI_Wtime();
    t_past_local += (t2 - t1);
#endif
                if ( recursiveChecker )
                {
                    if (locally_incomputable_col_idx >= myRowStart && locally_incomputable_col_idx<= myRowEnd)
                    {
                        last_result_val = k_level_result[ (my_level - 1)* myNumRow   + 
                                                    locally_incomputable_col_idx - 
                                                    myRowStart];
                    }
                    else
                    {
                        last_result_val = remoteValResultRecoder[locally_incomputable_col_idx][my_level - 1];

                    }
                    k_level_result[resPtr] += compactedCSR.csrdata[idx] * last_result_val;
                }

                recursiveChecker = false;
            }
        }
        else 
        {
            continue;
        }

#ifdef CAMPK_PROF_AFTER_COMM_COMPUT_FUNC_1  
    printf ("resPtr: %ld, RecursiveDependentEleComputation overhead: %lf sec\n", resPtr, t_past_local);
    t_past_localtotal += t_past_local;
    t_past_local = 0.0;
#endif        
    }
#ifdef CAMPK_PROF_AFTER_COMM_COMPUT_FUNC_1  
    printf ("RecursiveDependentEleComputation total overhead: %lf sec\n", t_past_localtotal);
#endif   
}

/////////////////////////////////////////////////////////////
void campk_after_comm_computation_v2 (csrType_local_var compactedCSR, double *k_level_result, short  *k_level_locally_computable_flags 
                                 , long vec_result_length, long myNumRow, long myRowStart, long myRowEnd,long *vec_remote_recv_idx 
                                 , double * buffer_vec_remote_recv, int numRemoteVec, int kval, int myid, int numprocs)
{

    long locally_incomputable_col_idx;
    long remoteEleStartIdx, remoteEleEndIdx;
    long remoteEleIdx;
    long remoteLocallyComputableColIdx;
    // std::map<long, std::vector<double> > remoteValResultRecoder;

    double * remoteValResultRecoderZone = (double *)malloc( compactedCSR.num_cols *(kval-1)*sizeof(double) );
    // set values of remoteValResultRecoderZone to be SPEC_VAL
    //, and assume that if a value of remoteValResultRecoderZone has changed
    //, it will be different from SPEC_VAL
    std::fill( remoteValResultRecoderZone, remoteValResultRecoderZone + compactedCSR.num_cols *(kval-1), SPEC_VAL );

    long idx;
    long eleStartIdx, eleEndIdx;
    long resPtr, myRowIdx;

    int my_level;
    int num_level_computable;

    double last_result_val;

    bool recursiveChecker=false;


    for (idx=0; idx<numRemoteVec;idx++)
    {
        remoteValResultRecoderZone[ vec_remote_recv_idx[idx] ] = buffer_vec_remote_recv[idx];
    }


    for (resPtr=myNumRow; resPtr<vec_result_length; resPtr++)
    {
        num_level_computable =  (int)k_level_locally_computable_flags[resPtr];

        my_level = (int) (resPtr / myNumRow);

        if (num_level_computable != my_level)
        {
            myRowIdx = (resPtr%myNumRow) + myRowStart;
            eleStartIdx = compactedCSR.row_start[myRowIdx];
            eleEndIdx   = compactedCSR.row_end[myRowIdx];

            for (idx = eleStartIdx; idx<= eleEndIdx; idx++)
            {
                locally_incomputable_col_idx = compactedCSR.col_idx[idx];

//////////////////////////////////// actural code starts
                recursiveChecker = RecursiveDependentEleComputation_mut(remoteValResultRecoderZone, compactedCSR, 
                            locally_incomputable_col_idx, my_level, k_level_result, myRowStart, myRowEnd, myNumRow,
                            myid, numprocs );
//////////////////////////////////// actural code ends
                if ( recursiveChecker )
                {
                    if (locally_incomputable_col_idx >= myRowStart && locally_incomputable_col_idx<= myRowEnd)
                    {
                        last_result_val = k_level_result[ (my_level - 1)* myNumRow   + 
                                                    locally_incomputable_col_idx - 
                                                    myRowStart];
                    }
                    else
                    {
                        last_result_val = remoteValResultRecoderZone[(my_level - 1)*compactedCSR.num_cols + locally_incomputable_col_idx];

                    }
                    k_level_result[resPtr] += compactedCSR.csrdata[idx] * last_result_val;
                }

                recursiveChecker = false;
            }
        }
        else 
        {
            continue;
        }
     
    }

    free(remoteValResultRecoderZone); 

}
/////////////////////////////////////////////////////////////

