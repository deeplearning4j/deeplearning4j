//
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/sequence_mask.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void sequenceMask(NDArray<T>* input, NDArray<T>* output, int maxIndex) {
#pragma omp parallel for if(maxIndex > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
        for (Nd4jIndex i = 0; i < maxIndex; i++)
            for(Nd4jIndex k = 0; k < input->lengthOf(); k++)
                if (i < static_cast<int>((*input)(k)))
                    (*output)(k * maxIndex + i) = T(1.0);
    }

    template void sequenceMask(NDArray<float>* input, NDArray<float>* output, int maxIndex);
    template void sequenceMask(NDArray<float16>* input, NDArray<float16>* output, int maxIndex);
    template void sequenceMask(NDArray<double>* input, NDArray<double>* output, int maxIndex);
}
}
}