//
//  @author sgazeos@gmail.com
//
#ifndef __LRN_H_HELPERS__
#define __LRN_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int lrnFunctor(NDArray<T>* input, NDArray<T>* output, int depth, T bias, T alpha, T beta);
    template <typename T>
    int lrnFunctorEx(NDArray<T>* input, NDArray<T>* output, NDArray<T>* unitScale, NDArray<T>* scale, int depth, T bias, T alpha, T beta);

}
}
}
#endif
