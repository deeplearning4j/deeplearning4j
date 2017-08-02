//
// @author raver119@gmail.com
//

#ifndef PROJECT_DENSE_H
#define PROJECT_DENSE_H

#include <layers/layers.h>
#include <layers/generic/BaseLayer.h>

namespace nd4j {
namespace layers {

template<typename T, typename AF> class DenseLayer: public BaseLayer<T, AF> {
    public:

        // default constructor
        DenseLayer();     

        // feed forward
        void feedForward();

        // back propagate
        void backPropagate();
       
        // This method should validate layer parameters & bias, and return TRUE if everything ok. FALSE otherwise      
        inline bool validateParameters();

        // This method should validate input parameters, and return TRUE if everything ok. FALSE otherwise
        inline bool validateInput();

        // This method should valudate output parameters, and return TRUE if everything is ok, FALSE otherwise        
        inline bool validateOutput();
};

/////// implementation part ///////    

// default constructor
template<typename T, typename AF> DenseLayer<T,AF>::DenseLayer() { 

}     

// back propagate
template<typename T, typename AF> void DenseLayer<T,AF>::backPropagate() {
    // activation derivative call
    ActivationsExecutioner<T>::template executeBP<AF>(this->input, this->epsilon, this->output, this->inputShapeInfo);
}


// This method should validate layer parameters & bias, and return TRUE if everything ok. FALSE otherwise
template<typename T, typename AF> bool DenseLayer<T,AF>::validateParameters() {
    // to be implemented
    return true;
}


// This method should validate input parameters, and return TRUE if everything ok. FALSE otherwise
template<typename T, typename AF> bool DenseLayer<T,AF>::validateInput() {
    // we expect input to be either vector or matrix, in both cases - that's rank2
    if (this->input == nullptr || this->inputShapeInfo == nullptr && shape::rank(this->inputShapeInfo) != 2)
        return false;

    return true;
}


// This method should valudate output parameters, and return TRUE if everything is ok, FALSE otherwise
template<typename T, typename AF> bool DenseLayer<T,AF>::validateOutput() {
    // same as input validation here. we expect rank of output arra
    if (this->output == nullptr || this->outputShapeInfo == nullptr && shape::rank(this->outputShapeInfo) != 2)
        return false;

    // length of output along dimension 1 should match length of parameters, if parameters are set,
    if (this->bias != nullptr) {

    }

    return true;
}

// feed forward
template<typename T, typename AF> void DenseLayer<T,AF>::feedForward() {
    // dropout helper call
    if (this->dropOut)
        dropOutHelper(this->input, this->inputShapeInfo);

    // dropconnect helper
    if (this->dropConnect)
        dropConnectHelper(this->params, this->paramsShapeInfo);

    int *inputShape = shape::shapeOf(this->inputShapeInfo);

    // do wxa+b here or something else
    // TODO: introduce BLAS right here
    if (shape::isRowVector(this->inputShapeInfo)) {
        // gemv here input * W
    } 
    else {
        // gemm here, input * W
        // these values should be set appropriately

        this->gemmHelper(this->input, this->inputShapeInfo, this->params, this->paramsShapeInfo, this->output, this->outputShapeInfo, (T) 1.0f, (T) 0.0f);

        // we're rolling through rows here
        int rowLen = this->outputShapeInfo[2];
        //#pragma omp parallel for
        for (int r = 0; r < this->outputShapeInfo[1]; r++) {
            T *row = this->output + (rowLen * r);

            // now we're adding bias to each row element
            //#pragma omp simd
            for (int e = 0; e < rowLen; e++) {
                row[e] += this->bias[e];
            }
        }
    }

    // activation call
    ActivationsExecutioner<T>::template executeFF<AF>(this->input, this->output, this->inputShapeInfo);
}


// end of namespace brackets
}
}

#endif //PROJECT_DENSE_H
