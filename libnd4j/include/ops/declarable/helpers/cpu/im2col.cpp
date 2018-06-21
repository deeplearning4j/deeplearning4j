//
// Created by raver119 on 30.11.17.
//

#include <ops/declarable/helpers/im2col.h>


namespace nd4j    {
namespace ops     {
namespace helpers {

// input [bS, iC, iH, iW] is convoluted to output [bS, iC, kH, kW, oH, oW]
template <typename T>
void _im2col(nd4j::graph::LaunchContext& context, T *col, T *im, Nd4jLong *colShape, Nd4jLong *imShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, T zeroPadVal) {
       
    T extraParams[] = {(T)kH, (T)kW, (T)sH, (T)sW, (T)pH, (T)pW, (T)dH, (T)dW, (T)isSameMode, zeroPadVal};

    functions::transform::Transform<T>::template exec<simdOps::Im2col<T>>(im, imShape, col, colShape, extraParams, nullptr, nullptr);    
}

template void _im2col<float>(nd4j::graph::LaunchContext& context, float *output, float *in, Nd4jLong *zShape, Nd4jLong *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, float zeroPadVal);
template void _im2col<float16>(nd4j::graph::LaunchContext& context, float16 *output, float16 *in, Nd4jLong *zShape, Nd4jLong *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, float16 zeroPadVal);
template void _im2col<double>(nd4j::graph::LaunchContext& context, double *output, double *in, Nd4jLong *zShape, Nd4jLong *xShape, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, double zeroPadVal);


}
}
}