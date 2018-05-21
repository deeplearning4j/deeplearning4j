//
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/helpers.h>
#ifndef __HELPERS__ROLL__H__
#define __HELPERS__ROLL__H__
namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    void rollFunctorLinear(NDArray<T>* input, NDArray<T>* output, int shift, bool inplace = false);

    template <typename T>
    void rollFunctorFull(NDArray<T>* input, NDArray<T>* output, int shift, std::vector<int> const& axes, bool inplace = false);
}
}
}
#endif
