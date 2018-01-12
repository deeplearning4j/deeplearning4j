//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/helpers.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    void _depthToSpace(NDArray<T> *input, NDArray<T> *output, int block_size, bool isNHWC);
}
}
}