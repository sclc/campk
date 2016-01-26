#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifndef MPI_HEADERS
#include <mpi.h>
#define MPI_HEADERS
#endif

#include "DataTypes.h"
#include "common.h"
#include "matrixType.h"
#include "GetRHS.h"
#include "mpk.h"
#include "campk.h"

// #define PROF_SPMM_v1
// #define PROF_SPMM_v2
// #define CAMPK_V1

#define COMPARAE_RES_1
// #define COMPARAE_RES_2

// #define PROF_RECURSIVE

#define PROF_ALL
#define COMPARAE_RES_1_MPK_PROF

// #define DB_COMPARAE_RES_1

int main(int argc, char* argv[]) {

    int myid, numprocs;
    long solverIdx;
    long sVal;
    char * path;
    char * mtx_filename;
    char * rhs_filename;
    char * string_num_cols;
    char * str_sVal;

// #pragma omp parallel
// {
//     omp_set_num_threads (2);
//     int ompnum = omp_get_num_threads();
//     int ompid = omp_get_thread_num();
//     printf ("total:%d, I am: %d\n", ompnum, ompid);
// }
#ifdef PROF_ALL
    double t1, t2, t_past_local, t_past_global;
#endif /*PROF_ALL*/

    if (argc < 6) {
        printf("Argument setting is wrong.\n");
        exit(0);
    } else {
        path         = argv[1];
        mtx_filename = argv[2];
        rhs_filename = argv[3];
        string_num_cols = argv[4];
        str_sVal = argv[5];
    }

    long set_num_cols;
    set_num_cols = (long)atoi(string_num_cols);

    sVal = (long)atoi(str_sVal);

    int ierr;


    // ierr = MPI_Init(&argc, &argv);
    int hybridProvided;
    printf ("MPI_THREAD_MULTIPLE:%d, MPI_THREAD_FUNNELED:%d, MPI_THREAD_SERIALIZED:%d \n", MPI_THREAD_MULTIPLE, MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED);
    ierr = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &hybridProvided);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);   
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    printf ("total: %d procs, myid:%d, I want:%d, I got:%d \n", numprocs, myid, MPI_THREAD_MULTIPLE, hybridProvided);

    if (myid == 0) {

        printf ("myid:%d, numprocs: %d\n", myid, numprocs);
        printf("we can want to generate RHS with size of %d\n", set_num_cols);
    }
    
    csrType_local local_Mat;

    denseType X;
    matInfo mat_info;

//////////////////////////////////////////////////////////////////////////
#ifdef COMPARAE_RES_1
    csrType_local local_Mat_comparator;
    Sparse_Csr_Matrix_Distribution(&local_Mat_comparator, &mat_info, myid, numprocs, path, mtx_filename);

    // GenVectorOne(mat_info.num_rows, &X, set_num_cols,myid, numprocs);
    GenVectorRandom(mat_info.num_rows, &X, set_num_cols,0.0 , 1.0, myid, numprocs);

    printf ("myid: %d. X.local_num_row: %ld,  X.local_num_col: %ld\n", myid, X.local_num_row, X.local_num_col);
    ierr = MPI_Barrier(MPI_COMM_WORLD);

    denseType AkX_compartor;

    gen_dense_mat(&AkX_compartor, sVal * X.local_num_row, 1
            , sVal * X.global_num_row, 1
            , local_Mat_comparator.start);

#ifdef COMPARAE_RES_1_MPK_PROF
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
#endif /*COMPARAE_RES_1_MPK_PROF*/
///////////////////////actural code start 
    mpk_v1(AkX_compartor, local_Mat_comparator, X, sVal, 1.0, 0.0, myid, numprocs);
///////////////////////actural code end
#ifdef COMPARAE_RES_1_MPK_PROF
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    t_past_local = t2 - t1;
    ierr = MPI_Reduce(&t_past_local, &t_past_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myid == 0)
    {
        printf ("mpk_v1 overhead: %lf sec\n", t_past_global);
    }
    printf ("myid: %d. mpk_v1 done\n", myid);
#endif /*COMPARAE_RES_1_MPK_PROF*/
    // ierr = MPI_Barrier(MPI_COMM_WORLD);
    // exit(1);
    
//  free memory space
    delete_csrType_local (local_Mat_comparator);

#ifdef DB_COMPARAE_RES_1
    ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif /*DB_COMPARAE_RES_1*/

    denseType AkX;
   //campk_v1(X, AkX, (int)sVal, &mat_info, myid, numprocs, path, mtx_filename);
   campk_v2(X, AkX, (int)sVal, &mat_info, myid, numprocs, path, mtx_filename);
#ifdef DB_COMPARAE_RES_1
    printf ("myid: %d. campk_v1 done\n", myid);
    ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif /*DB_COMPARAE_RES_1*/
    
    DenseMatrixComparsion (AkX, AkX_compartor);
    printf ("myid: %d. passed \n", myid);
    ierr = MPI_Barrier(MPI_COMM_WORLD);

#endif  /*COMPARAE_RES_1*/
//////////////////////////////////////////////////////////////////////////

///////////////////////actural code start 
#ifdef PROF_RECURSIVE
    int recursiveProfingIdx, recursiveProfingIdxTimes=100;
    int numLevel=1000;
    t_past_local = 0.0;

    for (recursiveProfingIdx=0; recursiveProfingIdx<recursiveProfingIdxTimes; recursiveProfingIdx++)
    {
        t1 = MPI_Wtime();
        RecurisiveCallOverheadProfiling (numLevel);       
        t2 = MPI_Wtime();
        t_past_local += t2 - t1;
    }
    
    printf ("myid: %d. recursive call overhead: %20.18lf\n", myid, t_past_local/recursiveProfingIdxTimes);
#endif /* PROF_RECURSIVE */
///////////////////////actural code end

///////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
#ifdef COMPARAE_RES_2
    double bandwidthRatio;
    bandwidthRatio = 0.75;
    long dim = 1024

    csrType_local local_Mat_comparator;
    ParallelGen_RandomNonSymetricSparseMatrix_CSR(&local_Mat_comparator, dim, bandwidthRatio);
    mat_info.num_rows  = dim;
    mat_info.num_cols  = dim;
    mat_info.nnz = 

    // GenVectorOne(mat_info.num_rows, &X, set_num_cols,myid, numprocs);
    GenVectorRandom(mat_info.num_rows, &X, set_num_cols,0.0 , 1.0, myid, numprocs);

    printf ("myid: %d. X.local_num_row: %ld,  X.local_num_col: %ld\n", myid, X.local_num_row, X.local_num_col);
    ierr = MPI_Barrier(MPI_COMM_WORLD);

    denseType AkX_compartor;

    gen_dense_mat(&AkX_compartor, sVal * X.local_num_row, 1
            , sVal * X.global_num_row, 1
            , local_Mat_comparator.start);

#ifdef MPK_PROF
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
#endif /*MPK_PROF*/
///////////////////////actural code start 
    mpk_v1(AkX_compartor, local_Mat_comparator, X, sVal, 1.0, 0.0, myid, numprocs);

///////////////////////actural code end
#ifdef MPK_PROF
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    t_past_local = t2 - t1;
    ierr = MPI_Reduce(&t_past_local, &t_past_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myid == 0)
    {
        printf ("mpk_v1 overhead: %lf sec\n", t_past_global);
    }
#endif /*MPK_PROF*/

#ifdef DB_COMPARAE_RES_1
    printf ("myid: %d. mpk_v1 done\n", myid);
    ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif /*DB_COMPARAE_RES_1*/

    denseType AkX;
    campk_v1(X, AkX, (int)sVal, &mat_info, myid, numprocs, path, mtx_filename);
#ifdef DB_COMPARAE_RES_1
    printf ("myid: %d. campk_v1 done\n", myid);
    ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif /*DB_COMPARAE_RES_1*/
    
    DenseMatrixComparsion (AkX, AkX_compartor);
    printf ("myid: %d. passed \n", myid);
    ierr = MPI_Barrier(MPI_COMM_WORLD);

#endif /*COMPARAE_RES_2*/
//////////////////////////////////////////////////////////////////////////
    // generate X
       // GenVectorOne(mat_info.num_rows, &B, set_num_cols,myid, numprocs);
    // GenVector_ReadCSV(&X, mat_info.num_rows, set_num_cols, rhs_filename, myid, numprocs);



#ifdef CAMPK_V1

    campk_spmv_v1(local_Mat, X, AkX, sVal, myid, numprocs);

#endif

#ifdef PROF_SPMM_v1
    double time_comm, time_computation, time_comm_sum=0.0, time_computation_sum=0.0;
    int loops = 1, loopIdx;

    ierr = MPI_Barrier(MPI_COMM_WORLD);
    if (myid==0)
    {
        printf ("start testing spmm_csr_v2, %d loops\n", loops);
    }
    
    for (loopIdx=0 ;loopIdx<loops; loopIdx++)
    {

        spmm_csr_v2_profiling(local_Mat, X, &B, myid, numprocs, &time_comm, &time_computation);
        ierr = MPI_Barrier(MPI_COMM_WORLD);

        time_comm_sum += time_comm;
        time_computation_sum += time_computation;
        
        if (myid==0)
        {
            printf ("loop %d, time_comm_sum = %lf, time_computation_sum = %lf\n",\
                     loopIdx, time_comm_sum, time_computation_sum);
        }
        
    }
    ierr = MPI_Barrier(MPI_COMM_WORLD);

    if (myid==0)
    {
        printf ("average, time_comm_sum = %lf, time_computation_sum = %lf\n",\
                time_comm_sum/(double)loops, time_computation_sum/(double)loops);
    }
    
#endif /*PROF_SPMM_v1*/

    //Garbage collection
    // delete_denseType(X);
    // do not forget to clean csr matrix
    // do not forget to free AkX[sVal]

    ierr = MPI_Finalize();

    return 0;

}
