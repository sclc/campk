#include "common.h"

// #ifndef ASSERTION_DEBUG
// #define ASSERTION_DEBUG
// #endif

char * concatStr(char * s1, char * s2) {
    char *resultStr = (char*)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(resultStr, s1);
    strcat(resultStr, s2);
    return resultStr;
}

//#define parseCSV_DEBUG

void parseCSV(char* filename, double** output, long numRows, long numCols) {
    long buffersize = 1024;
    char buf[buffersize];
    // make sure we have buf long enough for reading each line
    assert((sizeof (double) + sizeof (char)) * numCols < buffersize);

    FILE * fstream = fopen(filename, "r");
    assert(fstream != 0);
    *output = (double*) calloc(numRows*numCols, sizeof (double));

    long rowCounter = 0;
    const char * tok;
    long outputCounter = 0;
    while (rowCounter < numRows && fgets(buf, sizeof (buf), fstream)) {
        //        printf("%s", buf);
        // parse a line of CSV
        long rowEleCounter = 0;
        tok = strtok(buf, ",");
        //        printf("%s\n",tok);
        for (; tok&& *tok; tok = strtok(NULL, ",\n")) {
            // remember *output must be parenthesised
            (*output)[outputCounter++] = atof(tok);
            rowEleCounter++;
        }
        
        assert(rowEleCounter == numCols);

        rowCounter++;
    }


    fclose(fstream);
}

void Local_Dense_Mat_Generator(denseType * mat, long num_rows, long num_cols,\
                     double ranMin, double ranMax)
{
    srand (time(NULL));
    double ranRange = ranMax - ranMin;
    long totalEle = num_rows * num_cols;
    mat->data = (double*) calloc (totalEle,sizeof(double));

    long idx;
    for (idx=0; idx<totalEle; idx++)
    {
        mat->data[idx] = ranMin + ( (double)rand() / (double)RAND_MAX ) * ranRange;
    }

}

void dense_entry_copy_disp(denseType src, long srcDispStart, denseType target, long tarDispStart, long count) 
{

    long srcIdx, tarIdx;
    long srcEndIdx = srcDispStart + count;
    tarIdx = tarDispStart;
    for (srcIdx = srcDispStart; srcIdx < srcEndIdx; srcIdx++) {

        target.data[tarIdx] = src.data[srcIdx];
        tarIdx++;
    }
}


#define dense_mat_mat_add_TP_targetDisp_ASSERTION

void dense_mat_mat_add_TP_targetDisp(denseType mat1, denseType mat2, denseType output_mat, \
            long output_mat_disp, double alpha, double beta, int myid, int numprocs) 
{

    long num_rows = mat1.local_num_row;
    long num_cols = mat1.local_num_col;


    long idx_i;
    long idx_j;
    for (idx_i = 0; idx_i < num_rows; idx_i++) {
        for (idx_j = 0; idx_j < num_cols; idx_j++) {
            long mat_idx = idx_i * num_cols + idx_j;
#ifdef dense_mat_mat_add_TP_targetDisp_ASSERTION
            assert((output_mat_disp + mat_idx)< (output_mat.local_num_col * output_mat.local_num_row));
#endif
            output_mat.data[ output_mat_disp + mat_idx] =
                    alpha * mat1.data[mat_idx] + beta * mat2.data[mat_idx];
        }
    }
}

void dense_array_mat_mat_mat_add_TP_disp(double* mat1Data, long mat1Disp
                                       , double* mat2Data, long mat2Disp
                                       , double* mat3Data, long mat3Disp
                                       , double* output_mat, long output_mat_disp
                                       , long length
                                       , double alpha, double beta, double gamma
                                       , int myid, int numprocs){
    
    long idx;
    
    for (idx=0; idx<length;idx++){
        output_mat[output_mat_disp+idx] = alpha * mat1Data[mat1Disp+idx] 
                                        + beta  * mat2Data[mat2Disp+idx]
                                        + gamma * mat3Data[mat3Disp+idx];
    }
    
}


bool RecursiveDependentEleComputation(std::map<long, std::vector<double> >& remoteValResultRecoder, \
                                      csrType_local_var mat, long  eleIdx, int my_level,\
                                      double *k_level_result, long myRowStart, long myRowEnd, long myNumRow,
                                      int myid, int numprocs) 
 {
#ifdef DB_RDEC
    int ierr;
#endif
    long eleStartIdx, eleEndIdx;
    long idx;
    long dependentColIdx;
    double last_result_val;

    assert (my_level > 0);
    my_level--;

#ifdef ASSERTION_DEBUG
    assert (remoteValResultRecoder[eleIdx].size() - 1 >= 0);
#endif

#ifdef DB_RDEC_1_2
    std::cout<< "myid: "<<myid<<" , my_level: "<<my_level<<", eleIdx: "<<eleIdx<<std::endl;
    ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif

    // to avoid redundant computation
    if (eleIdx >= myRowStart && eleIdx <= myRowEnd) 
    {
        return true;
    } 
    else if ( remoteValResultRecoder[eleIdx].size() - 1 >= my_level)  
    {
#ifdef DB_RDEC_1_1
    // std::cout<< "myid: "<<myid<<" , my_level: "<<my_level<<", eleIdx: "<<eleIdx<<std::endl;
    // ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif
        return true;
    }
#ifdef DB_RDEC_1_2
    if (my_level==0)
    {
        std::cout<< "myid: "<<myid<<", my_level==0, eleIdx: "<<eleIdx<<" , ";
        std::cout<<"remoteValResultRecoder[eleIdx].size() - 1: "<<remoteValResultRecoder[eleIdx].size() - 1<<std::endl;
        ierr = MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
    }
#endif

    // all code later on are for eleIdx < myRowStart || eleIdx > myRowEnd cases
    eleStartIdx = mat.row_start[eleIdx];
    eleEndIdx   = mat.row_end  [eleIdx];

    // recursively compute remote dependent elements
    remoteValResultRecoder[eleIdx].push_back(0.0);

    for (idx = eleStartIdx; idx<=eleEndIdx; idx++)
    {
        dependentColIdx = mat.col_idx[idx];

        if ( RecursiveDependentEleComputation(remoteValResultRecoder, mat, dependentColIdx, my_level,\
                                    k_level_result, myRowStart, myRowEnd, myNumRow, myid, numprocs) )
        {
            if (dependentColIdx >= myRowStart && dependentColIdx <= myRowEnd)
            {
                last_result_val = k_level_result[ (my_level - 1)* myNumRow + dependentColIdx - myRowStart];
            }
            else
            {
#ifdef DB_RDEC_1_3
    if (remoteValResultRecoder[dependentColIdx].size() < my_level)
    {
        std::cout<< "myid: "<<myid<<", dependentColIdx: "<<dependentColIdx<<", my_level: "<<my_level<<", ";
        std::cout<<"remoteValResultRecoder[dependentColIdx].size(): "<<remoteValResultRecoder[dependentColIdx].size();
        std::cout<<std::endl;
        ierr = MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
    }
    // assert (remoteValResultRecoder[dependentColIdx].size() >= my_level);
#endif

#ifdef ASSERTION_DEBUG
    assert (remoteValResultRecoder[dependentColIdx].size() >= my_level);

#endif

#ifdef DB_RDEC_1_4
    if (myid == 1)
    {
        std::cout<< "myid: "<<myid<<", eleIdx:"<<eleIdx<<", dependentColIdx: "<<dependentColIdx<<", my_level: "<<my_level<<", ";
        std::cout<<"remoteValResultRecoder[dependentColIdx][my_level-1]: "<<remoteValResultRecoder[dependentColIdx][my_level-1];
        std::cout<<std::endl;
    }

#endif
                last_result_val = remoteValResultRecoder[dependentColIdx][my_level-1];
            }
            remoteValResultRecoder[eleIdx].back() += mat.csrdata[idx] * last_result_val;            
        }
    }

    return true;
 }

// storage in outMat has been allocated, inMat and outMat have the same size
// inMat is row order matrix, and outMat is the col order matrix
// inMat is a local dense matrix
// outMat is a local dense matrix, all properties of outMat has been set up appropriately

void dense_matrix_local_transpose_row_order(denseType inMat, denseType outMat) {

    long num_col = inMat.local_num_col;
    long num_row = inMat.local_num_row;

    long rowIdx, colIdx;

    for (rowIdx = 0; rowIdx < num_row; rowIdx++) {
        for (colIdx = 0; colIdx < num_col; colIdx++) {
            outMat.data[colIdx * num_row + rowIdx] = inMat.data[rowIdx * num_col + colIdx];
        }
    }
}

void DenseMatrixComparsion (denseType mat1, denseType mat2)
{
    long length = mat1.local_num_row * mat1.local_num_col;
    assert (mat1.local_num_row == mat2.local_num_row);
    assert (mat1.local_num_col == mat2.local_num_col);

    long idx;

    for  (idx=0; idx<length; idx++)
    {
        if (mat1.data[idx] != mat2.data[idx] )
        {
            printf ("idx:%ld, mat1: %lf, mat2: %lf difference: %lf\n", idx, mat1.data[idx], mat2.data[idx], mat1.data[idx]-mat2.data[idx]);
        }
    }
}

void ClearQueue (std::queue<long> &rowIdxScanQueue)
{
    std::queue<long> empty;
    std::swap(rowIdxScanQueue, empty);
}

void RecurisiveCallOverheadProfiling (int level)
{
    if (level==0) return;
    RecurisiveCallOverheadProfiling(level-1);

}