//
//  @author sgazeos@gmail.com
//
#ifndef __REDUCE_NORM_H_HELPERS__
#define __REDUCE_NORM_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void reduceNorm1BP(NDArray<T>* input, NDArray<T>* epsilon, NDArray<T>* tempNorm, NDArray<T>* output, std::vector<int> const& axes);

    template <typename T>
    void reduceNorm2BP(NDArray<T>* input, NDArray<T>* epsilon, NDArray<T>* tempNorm, NDArray<T>* output, std::vector<int> const& axes);

    template <typename T>
    void reduceSquareNormBP(NDArray<T>* input, NDArray<T>* epsilon, NDArray<T>* tempNorm, NDArray<T>* output, std::vector<int> const& axes);

}
}
}
#endif
