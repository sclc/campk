//#define GETRHS_DEBUG

#include "GetRHS.h"
#include "DataTypes.h"

// #define GenVectorRandom_DB

// this can give a dense matrix in row-major
void GenVectorOne(long length, denseType * vector, long num_cols, int myid, int numprocs) {
    long idx;
    long local_length, local_length_normal;
    
    // int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, 
    //            MPI_Comm comm )
    MPI_Bcast((void*) &length, 1, MPI_LONG, 0, MPI_COMM_WORLD);
#ifdef GETRHS_DEBUG
    printf("in GetRHS.c, myid=%d, length=%d\n", myid, length);
#endif

    local_length_normal = length / numprocs;
    if (myid == numprocs - 1)
        local_length = length - (numprocs - 1) * local_length_normal;
    else
        local_length = local_length_normal;
#ifdef GETRHS_DEBUG
    printf("in GetRHS.c, myid = %d, local_length=%d\n", myid, local_length);
#endif
    vector->local_num_row = local_length;
    vector->local_num_col = num_cols; // only consider
    vector->global_num_row = length;
    vector->global_num_col = num_cols;

#ifdef GETRHS_DEBUG
    printf("in GetRHS.c, myid=%d, vector->local_num_col=%d\n", myid, vector->local_num_col);
#endif

    vector->data = (double *) calloc(vector->local_num_row * vector->local_num_col, sizeof (double));

    //    if (myid == numprocs -1){
    //        long start_idx = myid * local_length_normal * num_cols;
    //        long end_idx   =  length * num_cols;
    //#ifdef GETRHS_DEBUG
    //        printf ("in GetRHS.c, myid=%d, start_idx=%d, end_idx=%d\n", myid, start_idx, end_idx);
    //#endif
    //        for (idx = start_idx; idx < end_idx;idx++){
    //            vector->data[idx] = 1.0;
    //        }
    //    }
    //    else{
    //        long start_idx = myid * local_length * num_cols;
    //        long end_idx   = (myid + 1) * local_length * num_cols;
    //#ifdef GETRHS_DEBUG
    //        printf ("in GetRHS.c, myid=%d, start_idx=%d, end_idx=%d\n", myid, start_idx, end_idx);
    //#endif
    //        for (idx = start_idx; idx < end_idx; idx++){
    //            vector->data[idx] = 1.0;
    //        }
    //    }
    long local_num_element = vector->local_num_row * vector->local_num_col;
    for (idx = 0; idx < local_num_element; idx++) {
        vector->data[idx] = 1.0;
    }

    // based on the assumption of equal division of rows among processes 
    vector->start_idx = myid * vector->global_num_col * local_length_normal;

}

// this can give a dense matrix in row-major
void GenVectorRandom(long length, denseType * vector, long num_cols, \
                     double ranMin, double ranMax, int myid, int numprocs)
{

    srand (time(NULL));
    double ranRange = ranMax - ranMin;

    long idx;
    long local_length, local_length_normal;
    MPI_Bcast((void*) &length, 1, MPI_LONG, 0, MPI_COMM_WORLD);
#ifdef GETRHS_DEBUG
    printf("in GetRHS.c, myid=%d, length=%d\n", myid, length);
#endif

    local_length_normal = length / numprocs;
    if (myid == numprocs - 1)
        local_length = length - (numprocs - 1) * local_length_normal;
    else
        local_length = local_length_normal;
#ifdef GETRHS_DEBUG
    printf("in GetRHS.c, myid = %d, local_length=%d\n", myid, local_length);
#endif
    vector->local_num_row = local_length;
    vector->local_num_col = num_cols; // only consider
    vector->global_num_row = length;
    vector->global_num_col = num_cols;

#ifdef GETRHS_DEBUG
    printf("in GetRHS.c, myid=%d, vector->local_num_col=%d\n", myid, vector->local_num_col);
#endif

    vector->data = (double *) calloc(vector->local_num_row * vector->local_num_col, sizeof (double));

    long local_num_element = vector->local_num_row * vector->local_num_col;
    for (idx = 0; idx < local_num_element; idx++) {
        double ranTemp = ranMin + ( (double)rand() / (double)RAND_MAX ) * ranRange;  
        vector->data[idx] = ranTemp;
    }

    // based on the assumption of equal division of rows among processes 
    vector->start_idx = myid * vector->global_num_col * local_length_normal;
#ifdef GenVectorRandom_DB

    if (myid == numprocs-1) {
        printf("start printing ...\n");
        local_dense_mat_print(*vector, myid);
    }
    exit(0);
#endif   

}

//#define GenVector_ReadCSV_DB

void GenVector_ReadCSV(denseType * vector, long length, long num_cols, char* rhsFile, int myid, int numprocs) {
    long idx;
    long local_length, local_length_normal;
    int ierr;
    double * Total_data_buffer;

    int sendCount[numprocs];
    int sendDispls[numprocs];

    long procCounter;
    double normel_ele_num;

    ierr = MPI_Bcast((void*) &length, 1, MPI_LONG, 0, MPI_COMM_WORLD);

#ifdef GETRHS_DEBUG
    printf("in GetRHS.c, myid=%d, length=%d\n", myid, length);
#endif

    local_length_normal = length / numprocs;
    if (myid == numprocs - 1)
        local_length = length - (numprocs - 1) * local_length_normal;
    else
        local_length = local_length_normal;
    normel_ele_num = local_length_normal * num_cols;
    for (procCounter = 0; procCounter < numprocs; procCounter++) {
        sendCount[procCounter] = (int)normel_ele_num;
        sendDispls[procCounter] = (int)procCounter * normel_ele_num;
    }
    sendCount[numprocs - 1] = (int)((length - (numprocs - 1) * local_length_normal) * num_cols);

#ifdef GETRHS_DEBUG
    printf("in GetRHS.c, myid = %d, local_length=%d\n", myid, local_length);
#endif
    vector->local_num_row = local_length;
    vector->local_num_col = num_cols; // only consider
    vector->global_num_row = length;
    vector->global_num_col = num_cols;

#ifdef GETRHS_DEBUG

#endif

    vector->data = (double *) calloc(vector->local_num_row * vector->local_num_col, sizeof (double));

    long local_num_element = vector->local_num_row * vector->local_num_col;

    // rank 0 read CSV
    if (myid == 0) {
        printf("Reading MRHS data from %s ... ...\n", rhsFile);
        parseCSV(rhsFile, &Total_data_buffer, length, num_cols);
        printf("Reading MRHS data from %s done.\n", rhsFile);
#ifdef GenVector_ReadCSV_DB
        //check_csv_array_print(Total_data_buffer, length, num_cols, myid);
        //        exit(0);
#endif

    }
    //    // Scatter data

// int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
//                  MPI_Datatype sendtype, void *recvbuf, int recvcount,
//                  MPI_Datatype recvtype,
//                  int root, MPI_Comm comm)

    ierr = MPI_Scatterv((void*) Total_data_buffer, (int*)sendCount, (int*)sendDispls,
            MPI_DOUBLE, vector->data, (int)local_num_element,
            MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //
    //    // based on the assumption of equal division of rows among processes 
    vector->start_idx = myid * vector->global_num_col * local_length_normal;
#ifdef GenVector_ReadCSV_DB
    //    if (myid == 0){
    //        check_csv_array_print(vector->data, vector->local_num_row, vector->global_num_col, myid);
    //        printf ("local rows:%d, local cols %d, local nnz:%d\n", vector->local_num_row, vector->local_num_col,local_num_element);
    //    }
    if (myid == numprocs-1) {
        local_dense_mat_print(*vector, myid);
    }
    exit(0);
#endif    
    if (myid == 0) {
        free(Total_data_buffer);
    }

}

