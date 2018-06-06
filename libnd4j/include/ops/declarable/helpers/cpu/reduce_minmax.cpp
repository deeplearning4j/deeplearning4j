//
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/axis.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void minMaxReduceFunctor(NDArray<T>* input, NDArray<T>* tempVals, NDArray<T>* output) {
            if (tempVals->isScalar()) {
                for (Nd4jLong e = 0; e < input->lengthOf(); e++) {
                    if (nd4j::math::nd4j_abs((*tempVals)(0) - (*input)(e)) < T(1.E-5f)) { // if input value equals to max
                         (*output)(e) = (*input)(e);
                    }
                }
            }
            else {
                for (Nd4jLong e = 0; e < input->lengthOf(); e++) {
                    for (Nd4jLong j = 0; j < tempVals->lengthOf(); j++) {
                        if (nd4j::math::nd4j_abs((*tempVals)(j) - (*input)(e)) < T(1.E-5f))  // if input value equals to max
                            (*output)(e) = (*input)(e);
                    }
                }
            }

    }

    template void minMaxReduceFunctor(NDArray<float>* input, NDArray<float>* tempVals, NDArray<float>* output);
    template void minMaxReduceFunctor(NDArray<float16>* input, NDArray<float16>* tempVals, NDArray<float16>*  output);
    template void minMaxReduceFunctor(NDArray<double>* input, NDArray<double>* tempVals, NDArray<double>* output);
}
}
}