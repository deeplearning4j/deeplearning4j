//
//  @author sgazeos@gmail.com
//
#ifndef __TOP_K_HELPERS__
#define __TOP_K_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int topKFunctor(NDArray<T>* input, NDArray<T>* values, NDArray<T>* indeces, int k, bool needSort);
    template <typename T>
    int inTopKFunctor(NDArray<T>* input, NDArray<T>* target, NDArray<T>* result, int k);

}
}
}
#endif
