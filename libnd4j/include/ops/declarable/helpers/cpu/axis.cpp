//
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/axis.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void adjustAxis(NDArray<T>* input, NDArray<T>* axisVector, std::vector<int>& output) {
            for (int e = 0; e < axisVector->lengthOf(); e++) {
                    int ca = (int) (*axisVector)(e);
                    if (ca < 0)
                        ca += input->rankOf();

                    output[e] = ca;
            }
    }

    template void adjustAxis(NDArray<float>* input, NDArray<float>* axisVector, std::vector<int>& output);
    template void adjustAxis(NDArray<float16>* input, NDArray<float16>* axisVector, std::vector<int>& output);
    template void adjustAxis(NDArray<double>* input, NDArray<double>* axisVector, std::vector<int>& output);
}
}
}