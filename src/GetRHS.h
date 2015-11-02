/* 
 * File:   GetRHS.h
 * Author: scl
 *
 * Created on December 11, 2013, 9:14 PM
 */

#ifndef MPI_HEADERS
#include <mpi.h>
#define MPI_HEADERS
#endif
 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "DataTypes.h"
#include "common.h"
    
#ifdef	__cplusplus
extern "C" {
#endif
	
void GenVectorOne(long length, denseType * vector, long num_cols,int myid, int numprocs);

void GenVectorRandom(long length, denseType * vector, long num_cols, \
	                 double ranMin, double ranMax, int myid, int numprocs);

// 1st: make sure csv file has enough elements, dense matrix size is given
// 2nd: rank0 read matrix, and then distribute MPI_Scatter

void GenVector_ReadCSV(denseType * vector, long length, long num_cols, char* rhsFile,int myid, int numprocs);


#ifdef	__cplusplus
}
#endif
