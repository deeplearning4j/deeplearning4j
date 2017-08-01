//
// @author raver119@gmail.com
//

#ifndef PROJECT_DENSE_H
#define PROJECT_DENSE_H

#include <layers/layers.h>
#include <layers/generic/BaseLayer.h>

namespace nd4j {
    namespace layers {

        template<typename T, typename AF>
        class DenseLayer: public BaseLayer<T, AF> {

            void feedForward() {
                // dropout helper call
                if (dropOut)
                    dropOutHelper(input, inputShapeInfo);

                // dropconnect helper
                if (dropConnect)
                    dropConnectHelper(params, paramshapeInfo);

                // do wxa+b here or something else
                // TODO: introduce BLAS right here
                if (shape::isRowVector(inputShapeInfo)) {
                    // gemv here input * W
                } else {
                    // gemm here, input * W
                }

                // activation call
                ActivationsExecutioner<T>::template executeFF<AF>(this->input, this->output, this->inputShapeInfo);
            }

            void backPropagate() {
                //

                // activation derivative call
                ActivationsExecutioner<T>::template executeBP<AF>(this->input, this->epsilon, this->output, this->inputShapeInfo);
            }
        };
    }
}

#endif //PROJECT_DENSE_H
