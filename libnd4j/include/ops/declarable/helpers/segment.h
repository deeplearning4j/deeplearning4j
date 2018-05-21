//
//  @author sgazeos@gmail.com
//  @brief helpers fuctions for segment_* ops (segment_max, segment_min, segment_mean, segment_sum and segment_prod)
//
#ifndef __SEGMENT_HELPERS__
#define __SEGMENT_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    bool segmentIndicesValidate(NDArray<T>* indices, T& expected, T& output);

    template <typename T>
    void segmentMaxFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output);

    template <typename T>
    void segmentMinFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output);

    template <typename T>
    void segmentMeanFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output);

    template <typename T>
    void segmentSumFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output);

    template <typename T>
    void segmentProdFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output);

}
}
}
#endif
