//
//  @author sgazeos@gmail.com
//
#ifndef __WEIGHTS_H_HELPERS__
#define __WEIGHTS_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void adjustWeights(NDArray<T>* input, NDArray<T>* weights, NDArray<T>* output, int minLength, int maxLength);

}
}
}
#endif
