//
//  @author sgazeos@gmail.com
//
#ifndef __REDUCE_DOT_H_HELPERS__
#define __REDUCE_DOT_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void reduceDotBP(NDArray<T>* inputX, NDArray<T>* inputY, NDArray<T>* epsilon, NDArray<T>* output, std::vector<int> const& axes);

}
}
}
#endif
