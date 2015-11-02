#include "MatGenerator.h"

#define DB_MAT_GEN_1

void TridiagonalCSR_val(csrType_local * mat, long rowDim, long colDim, double val)
{

	mat->num_rows = rowDim;
	mat->num_cols = colDim;
	mat->nnz      = 3 * rowDim - 2; 
	mat->start    = 0;

	mat->row_start = (long *)calloc( rowDim+1, sizeof(long) );
	mat->col_idx   = (long *)calloc ( mat->nnz, sizeof(long) );
	mat->csrdata   = (double *) calloc ( mat->nnz, sizeof(double) );


	long idx, lastRowIdx = mat->num_rows - 1;
	long rowStart;

	for (idx = 0; idx<mat->nnz; idx++)
	{
		mat->csrdata[idx] = val;
	}

	mat->col_idx[0] = 0;
	mat->col_idx[1] = 1;
	mat->row_start[0] = 0;

	mat->col_idx[mat->nnz - 2] = mat->num_cols - 2 ;
	mat->col_idx[mat->nnz - 1] = mat->num_cols - 1 ;
	mat->row_start[lastRowIdx] = mat->nnz - 2;
	mat->row_start[mat->num_rows] = mat->nnz;

	for (idx = 1; idx< lastRowIdx; idx++)
	{
		rowStart = 3 * idx - 1;
		mat->col_idx[rowStart]     = idx - 1;
		mat->col_idx[rowStart + 1] = idx;
		mat->col_idx[rowStart + 2] = idx + 1;

		mat->row_start[idx] =  rowStart;

	}


}

// a version gonna be used

long ParallelGen_RandomNonSymetricSparseMatrix_CSR(csrType_local * localMat \
                                            ,long dim, double offDiaHalfBandwidthRatio, int myid, int numprocs)
{
    int ierr;
    long idx;
    long totalNNZ = 0;
 
    srand (time(NULL));

    long num_row_local;

    long off_dia_half_bandwidth = (long) (dim * offDiaHalfBandwidthRatio);
    long mat_bandwidth = off_dia_half_bandwidth * 2 + 1;

    if (mat_bandwidth > 10000)
    {
        printf("%d,  Error: you want a sparse with too large bandwidth\n", myid);
    }
 
    if ((long)(dim/(long)numprocs) > 100000) 
    {
        printf ("%d, Error: you have more than 100000 rows per process\n", myid);
        exit(0);
    }

// if rows_per_proc<100000,  int is enough
// for my caksms project use, I have to keep the data consistency right now
    long rows_per_proc = dim / numprocs;


    if (myid == (numprocs - 1)) {
        num_row_local = dim - rows_per_proc * (numprocs - 1);
    } else {
        num_row_local = rows_per_proc;
    }

    localMat->num_rows = num_row_local;

    localMat->num_cols = dim;

    localMat->row_start = (long *) calloc(num_row_local + 1, sizeof (long));
    localMat->start = myid * rows_per_proc;

    // printf("good myid:%d \n",myid);
 
    long num_items_per_row_lower_bound = ITEMS_PER_ROW_LOWER_BOUND;
    long num_items_per_row_upper_bound = (off_dia_half_bandwidth + 1) * SPARSITY_RATIO;
    // long num_items_per_row_upper_bound = mat_bandwidth * SPARSITY_RATIO;
    long range_left_items_per_row = num_items_per_row_upper_bound - num_items_per_row_lower_bound;

    if (num_items_per_row_lower_bound > num_items_per_row_upper_bound)
    {
        printf("num_items_per_row_lower_bound > num_items_per_row_upper_bound \
               , you may need to change SPARSITY_RATIO to a larger value\n");
        exit(0);
    }
    long nnz_counter = 0;
    localMat->row_start[0] = 0; 

    for (idx = 1; idx <= num_row_local; idx++) {

        long num_items_last_row = num_items_per_row_lower_bound \
                               + ( (double)rand() / (double)RAND_MAX ) * range_left_items_per_row;
                            
        nnz_counter += num_items_last_row;

        localMat->row_start[idx] = localMat->row_start[idx-1] + num_items_last_row;
 
    }

    localMat->nnz = nnz_counter;

// int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
//                MPI_Op op, int root, MPI_Comm comm)
    // ierr = MPI_Reduce ( (void*) &nnz_counter, )  // unfinish

    localMat->col_idx = (long *) calloc(localMat->nnz, sizeof (long));
    localMat->csrdata = (double *) calloc(localMat->nnz, sizeof (double));

    long global_idx_row_start;

    if (myid == numprocs -1)
    {  
        global_idx_row_start = dim - localMat->num_rows; 
    }
    else
    {
        global_idx_row_start = myid * rows_per_proc;
    }

    long duplicate_check_lable = dim+10000; 

    double valUpper = VAL_UPPER;
    double valLower = VAL_LOWER; 
    double valRange = valUpper - valLower;

    for (idx = 0; idx < num_row_local; idx++) 
    {   
        long global_idx_this_row = global_idx_row_start + idx;

        long lower_off_dia_idx_max = global_idx_this_row - off_dia_half_bandwidth;
        long col_idx_lower_bound_this_row =  lower_off_dia_idx_max >= 0 ? \
                                            lower_off_dia_idx_max : 0;

        long upper_off_dia_idx_max = global_idx_this_row + off_dia_half_bandwidth;
        long col_idx_upper_bound_this_row = upper_off_dia_idx_max <= (dim - 1)? \
                                            upper_off_dia_idx_max: (dim-1);

        long range_col_idx = col_idx_upper_bound_this_row - col_idx_lower_bound_this_row + 1;
        if (range_col_idx<=0) 
        {
            printf ("when generating col idx, you set an odd range, check range_col_idx\n");
            exit(0);
        }

        long idx_nnz_this_row;
        long nnz_this_row = localMat->row_start[idx+1] - localMat->row_start[idx];

        // calloc should be malloc + memset
        long* col_idx_check_table = (long*) calloc (range_col_idx, sizeof(long));


        for (idx_nnz_this_row = 0; idx_nnz_this_row < nnz_this_row; idx_nnz_this_row++)
        {
            long col_off_displacement = (long)( (double)rand() / (double)RAND_MAX )* range_col_idx;

            long changer = 0;

            while (col_idx_check_table[col_off_displacement] == duplicate_check_lable) 
            {

                col_off_displacement = (long)( (double)rand() / (double)RAND_MAX )* range_col_idx;
                
                // hopely, this only happen when range_col_idx is very small,
                // so jump out of while loop, give up randomly generate col_off_displacement
                changer++;

                if(changer > CHANGER_THRESHOLD) 
                {
                    col_off_displacement = dim + 100000;
                    break;
                }
            }

            if (col_off_displacement == (dim + 100000))
            {
                long posSearch =0;
                while (col_idx_check_table[posSearch] == duplicate_check_lable)
                {
                    posSearch++;
                }
                col_off_displacement = posSearch;
            }

            // long col_idx_on_this_row = col_off_displacement + lower_off_dia_idx_max;
            long col_idx_on_this_row = col_off_displacement + col_idx_lower_bound_this_row;

            localMat->col_idx[localMat->row_start[idx] + idx_nnz_this_row] = col_idx_on_this_row;
            col_idx_check_table[col_off_displacement] = duplicate_check_lable;

            double this_val = valLower + (double) ((double)rand()/(double)RAND_MAX)* valRange;
            localMat->csrdata[localMat->row_start[idx] + idx_nnz_this_row] = this_val;


        }

    }

// free useless buffer
    return totalNNZ;
}