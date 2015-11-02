/* 
 * File:   DataTypes.h
 * Author: scl
 *
 * Created on November 27, 2013, 5:29 PM
 */

#ifndef DATATYPES_H
#define DATATYPES_H

#include <map>
#include <vector>

// #ifdef  __cplusplus
// extern "C" {
// #endif

typedef struct {

    long num_rows;
    long num_cols;
    long nnz;

} matInfo;

typedef struct
{
                 
    /* Dimensions, number of nonzeros 
     * (m == n for square, m < n for a local "slice") */
    // local matrix property
    long num_rows, num_cols, nnz;
                     
    /* Start of the rows owned by each thread */
    long start;      

    /* Starts of rows owned by local processor
     * row_start[0] == 0 == offset of first nz in row start[MY_THREAD] 
     */
    long *row_start;
                                          
    /* Column indices and values of matrix elements at local processor */
    long *col_idx;
    double *csrdata;
                                          
} csrType_local;

typedef struct
{
                 
    /* Dimensions, number of nonzeros 
     * (m == n for square, m < n for a local "slice") */
    // local matrix property
    long num_rows, num_cols, nnz;
                     
    /* Starts of rows owned by local processor
     * row_start[0] == 0 == offset of first nz in row start[MY_THREAD] 
     */
    std::map<long,long> row_start;
    std::map<long,long> row_end;
                                          
    /* Column indices and values of matrix elements at local processor */
    std::vector<long> col_idx;
    std::vector<double> csrdata;
                                          
} csrType_local_var;


typedef struct 
{
    long * rowIdx;
    long * colIdx;
    double * coodata;
} cooType;

typedef struct
{
    long local_num_row;
    long local_num_col;

    long global_num_row;
    long global_num_col;
    long start_idx;
        
    double * data;
} denseType;


typedef struct
{
                 

    long * rowIdxOnProcess;

    // it will have numporcs+1 elements
    //, which are pointing the header idx of each level on rowIdxOnProcess
    long * processPtr;
                     
                                          
} RowDistTable;

// #ifdef  __cplusplus
// }
// #endif

#endif  /* DATATYPES_H */
