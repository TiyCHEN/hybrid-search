#include <stddef.h>
typedef float (*SIMDFuncType)(const float *pv1, const float *pv2, size_t dim);
extern SIMDFuncType SIMDFunc;
void SetSIMDFunc();