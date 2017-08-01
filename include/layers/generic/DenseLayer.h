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
        public:

            DenseLayer() {
                //
            }

            void feedForward() {
                // dropout helper call
                if (this->dropOut)
                    dropOutHelper(this->input, this->inputShapeInfo);

                // dropconnect helper
                if (this->dropConnect)
                    dropConnectHelper(this->params, this->paramshapeInfo);

                // do wxa+b here or something else
                // TODO: introduce BLAS right here
                if (shape::isRowVector(this->inputShapeInfo)) {
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



             bool validateParameters() {
                 // to be implemented
                 return true;
             }

            /**
             * This method should validate input parameters, and return TRUE if everything ok. False otherwise
             * @return
             */
            bool validateInput() {
                // to be implemented
                return true;
            }

            /**
             * This method should valudate output parameters, and return TRUE if everything is ok, FALSE otherwise
             * @return
             */
            bool validateOutput() {
                // to be implemented
                return true;
            }
        };
    }
}

#endif //PROJECT_DENSE_H
