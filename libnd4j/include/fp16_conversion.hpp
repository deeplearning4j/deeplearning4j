#ifndef CAFFE_UTIL_FP16_CONVERSION_H_
#define CAFFE_UTIL_FP16_CONVERSION_H_

#ifndef CPU_ONLY

#include <cuda_fp16.h>
// Host functions for converting between FP32 and FP16 formats
// Paulius Micikevicius (pauliusm@nvidia.com)

half cpu_float2half_rn(float f);
float cpu_half2float(half h);
half operator - (const half& h);
int isnan(const half& h);
int isinf(const half& h);

#endif // CPU_ONLY

#endif
