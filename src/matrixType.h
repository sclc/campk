// define matrix types
#include "DataTypes.h"
#include <stdlib.h>


#ifdef	__cplusplus
extern "C" {
#endif

void Converter_Coo2Csr (cooType src, csrType_local * target, matInfo * mat_info);

void delete_csrType_local   (csrType_local mat);
void delete_cooType   (cooType mat);
void delete_denseType (denseType mat);

void get_same_shape_denseType (denseType src, denseType *target);
void set_dense_to_zero (denseType mat);

void gen_dense_mat (denseType *mat, long local_row_num, long local_col_num
                  , long global_row_num, long global_col_num, long start_row_id);

#ifdef	__cplusplus
}
#endif