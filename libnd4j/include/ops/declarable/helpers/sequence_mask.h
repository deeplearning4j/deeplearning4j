//
//  @author sgazeos@gmail.com
//
#ifndef __SEQUENCE_MASK_HELPERS__
#define __SEQUENCE_MASK_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void sequenceMask(NDArray<T>* input, NDArray<T>* output, int maxIndex);

}
}
}
#endif
