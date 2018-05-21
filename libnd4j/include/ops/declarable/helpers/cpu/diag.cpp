//
// Created by GS <sgazeos@gmail.com> on 4/6/2018.
//

#include "ResultSet.h"
#include "NDArrayFactory.h"
#include <ops/declarable/helpers/diag.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// Returns a batched matrix tensor with new batched diagonal values.
// for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/matrix_set_diag
template <typename T>
void diagFunctor(const NDArray<T>* input, NDArray<T>* output) {

    const int inLength = input->lengthOf();    

#pragma omp parallel for if(inLength > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
    for(int i = 0; i < inLength; ++i)
        (*output)(i * (inLength + 1)) = (*input)(i);
}


template void diagFunctor<float>(const NDArray<float>* input, NDArray<float>* output);
template void diagFunctor<float16>(const NDArray<float16>* input, NDArray<float16>* output);
template void diagFunctor<double>(const NDArray<double>* input, NDArray<double>* output);


}
}
}