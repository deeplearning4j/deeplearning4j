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
                    int M, N, K;
                    int lda, ldb, ldc;
                    T alpha, beta;

                    nd4j::blas::GEMM<T>::op('f', true, false, M, N, K, alpha, this->input, lda, this->params, ldb, beta, this->output, ldc);
                }

                // activation call
                ActivationsExecutioner<T>::template executeFF<AF>(this->input, this->output, this->inputShapeInfo);
            }

            void backPropagate() {
                //

                // activation derivative call
                ActivationsExecutioner<T>::template executeBP<AF>(this->input, this->epsilon, this->output, this->inputShapeInfo);
            }


            /**
             * This method should validate layer parameters & bias, and return TRUE if everything ok. FALSE otherwise
             *
             * @return
             */
             bool validateParameters() {
                 // to be implemented

                 return true;
             }

            /**
             * This method should validate input parameters, and return TRUE if everything ok. FALSE otherwise
             * @return
             */
            bool validateInput() {
                // we expect input to be either vector or matrix, in both cases - that's rank2
                if (this->input == nullptr || this->inputShapeInfo == nullptr && shape::rank(this->inputShapeInfo) != 2)
                    return false;

                return true;
            }

            /**
             * This method should valudate output parameters, and return TRUE if everything is ok, FALSE otherwise
             * @return
             */
            bool validateOutput() {
                // same as input validation here. we expect rank of output arra
                if (this->output == nullptr || this->outputShapeInfo == nullptr && shape::rank(this->outputShapeInfo) != 2)
                    return false;

                // length of output along dimension 1 should match length of parameters, if parameters are set,
                if (this->bias != nullptr) {

                }

                return true;
            }
        };
    }
}

#endif //PROJECT_DENSE_H
