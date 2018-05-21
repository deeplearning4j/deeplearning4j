//
//  @author sgazeos@gmail.com
//
#ifndef __MAX_POOLING_HELPERS__
#define __MAX_POOLING_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void maxPoolingFunctor(NDArray<T>* input, NDArray<T>* values, std::vector<int> const& params, NDArray<T>* indices);
}
}
}
#endif
