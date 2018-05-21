//
//  @author sgazeos@gmail.com
//
#ifndef __BDS_H_HELPERS__
#define __BDS_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int bdsFunctor(NDArray<T>* x_shape, NDArray<T>* y_shape, NDArray<T>* output);
}
}
}
#endif
