#include "DataTypes.h"
#include "common.h"
#include "mmio.h"
#include <assert.h>


#ifdef	__cplusplus
extern "C" {
#endif

void readMtx_coo(char* path, char* name, cooType mtr, matInfo info);
void readMtx_info_and_coo(char* path, char* name, matInfo* info, cooType* mat);

#ifdef	__cplusplus
}
#endif