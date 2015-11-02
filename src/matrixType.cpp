// 
#include "matrixType.h"

void Converter_Coo2Csr (cooType src, csrType_local * target, matInfo * mat_info)
{
    long idx, cumsum, last;
    
    target->num_rows = mat_info->num_rows;
    target->num_cols = mat_info->num_cols;
    target->nnz      = mat_info->nnz;
    
    target->row_start = (long *) calloc (target->num_rows + 1, sizeof(long) );
    target->col_idx   = (long *) calloc (target->nnz, sizeof (long));
    target->csrdata   = (double *) calloc (target->nnz, sizeof(double));
    
    for (idx = 0; idx < mat_info->num_rows; idx++)
        target->row_start[idx] = 0;

    for (idx = 0; idx < mat_info->nnz; idx++)
        target->row_start[src.rowIdx[idx]]++;


    //cumsum the nnz per row to get Bp[]
    for(idx = 0, cumsum = 0; idx < mat_info->num_rows; idx++){     
        long temp = target->row_start[idx];
        target->row_start[idx] = cumsum;
        cumsum += temp;
    }
    target->row_start[mat_info->num_rows] = mat_info->nnz;

    // final csr, in row section unsorted by column or essentially sorted for 
    // coo sorted by col
    for(idx = 0; idx < mat_info->nnz; idx++){
        long row  = src.rowIdx[idx];
        long dest = target->row_start[row];

        target->col_idx[dest] = src.colIdx [idx];
        target->csrdata[dest] = src.coodata[idx];
        // printf ("%lf\n", target->csrdata[dest]);

        target->row_start[row]++;
    }
    
    last = 0;
    // set back row_start values
    for( idx = 0; idx <= mat_info->num_rows; idx++){
        long temp = target->row_start[idx];
        target->row_start[idx]  = last;
        last   = temp;
    }
    
}

void delete_csrType_local  (csrType_local mat)
{
	free(mat.row_start);
	free(mat.col_idx);
	free(mat.csrdata);
}

void delete_cooType (cooType mat)
{
	free(mat.rowIdx) ;
	free(mat.colIdx) ;
	free(mat.coodata);
}

void delete_denseType (denseType mat)
{
    if (mat.data == 0){
        exit(0);
    }
    free (mat.data);
}

void get_same_shape_denseType (denseType src, denseType *target)
{
    target->local_num_col = src.local_num_col;
    target->local_num_row = src.local_num_row;
    target->global_num_col = src.global_num_col;
    target->global_num_row = src.global_num_row;
    target->start_idx      = src.start_idx;
    
    target->data = (double *)calloc( target->local_num_col * target->local_num_row, sizeof(double));
}

void set_dense_to_zero (denseType mat){
    long idx;
    long total_elements = mat.local_num_col * mat.local_num_row;
    for ( idx=0; idx < total_elements; idx++){
        mat.data[idx] = 0.0;
    }
}

void gen_dense_mat (denseType *mat, long local_row_num, long local_col_num
                  , long global_row_num, long global_col_num, long start_row_id){
    mat->global_num_col = global_col_num;
    mat->global_num_row = global_row_num;
    
    mat->local_num_col = local_col_num;
    mat->local_num_row = local_row_num;
    
    mat->start_idx = start_row_id;
    
    mat->data = (double*)calloc(local_row_num * local_col_num, sizeof(double));
}