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



    //this->gemmHelper(this->input, this->inputShapeInfo, delta, this->epsilonShapeInfo, this->output, this->outputShapeInfo, (T)1.0f, (T)0.0f);
    
    // INDArray delta = layerConf().getActivationFn().backprop(z, epsilon).getFirst();
    // how to evaluate delta, what is it ???
    // this->gemmHelper(this->input, this->inputShapeInfo, this->params, this->paramsShapeInfo, this->output, this->outputShapeInfo, (T) 1.0f, (T) 0.0f);
    // Nd4j.gemm(input, delta, weightGrad, true, false, 1.0, 0.0);

}







  // inline static T bpActivation(T value, T epsilon) {
                // // FIXME: ultra-bad. should consider conigurable extra params here
                // T extra[] = {(T) 0.0f};
                // return simdOps::Step<T>::template op(value, extra) * epsilon;





// This method should validate layer parameters & bias, and return TRUE if everything ok. FALSE otherwise
template<typename T, typename AF>
bool DenseLayer<T,AF>::validateParameters() {
    if (this->paramsShapeInfo == nullptr || this->biasShapeInfo == nullptr || this->params == nullptr || this->bias == nullptr) {
        printf("Got nulls here\n");
        return false;
    }

    int wRank = shape::rank(this->paramsShapeInfo);
    int bRank = shape::rank(this->biasShapeInfo);

    // rank of params/bias has to be 2 here
    if (wRank != 2 || bRank != 2) {
        printf("Non 2\n");
        return false;
    }


    int *wShape = shape::shapeOf(this->paramsShapeInfo);

    int biasLength = shape::length(this->biasShapeInfo);

    // number of outputs must be equal to biasLength
    if (wShape[1] != biasLength) {
        printf("Bias doesn't match: %i vs %i\n", wShape[1], biasLength);
        return false;
    }


    return true;
}


// This method should validate input parameters, and return TRUE if everything ok. FALSE otherwise
template<typename T, typename AF> bool DenseLayer<T,AF>::validateInput() {
    // we expect input to be either vector or matrix, in both cases - that's rank2
    if (this->input == nullptr || this->inputShapeInfo == nullptr || shape::rank(this->inputShapeInfo) != 2)
        return false;

    int *iShape = shape::shapeOf(this->inputShapeInfo);

    if (this->params != nullptr) {
        // check dimensionality

        int *wShape = shape::shapeOf(this->paramsShapeInfo);

        // number of input features should match number of rows in params
        if (iShape[1] != wShape[0]) {
            return false;
        }
    }

    if (this->output != nullptr) {
        int *oShape = shape::shapeOf(this->outputShapeInfo);

        // we check for input/output batchSize equality
        if (oShape[0] != iShape[0])
            return false;
    }


    return true;
}



// This method should valudate output parameters, and return TRUE if everything is ok, FALSE otherwise
template<typename T, typename AF> bool DenseLayer<T,AF>::validateOutput() {
    // same as input validation here. we expect rank of output arra
    if (this->output == nullptr || this->outputShapeInfo == nullptr || shape::rank(this->outputShapeInfo) != 2)
        return false;

    int *oShape = shape::shapeOf(this->outputShapeInfo);

    // length of output along dimension 1 should match length of parameters, if parameters are set,
    if (this->params != nullptr) {
        int *wShape = shape::shapeOf(this->paramsShapeInfo);

        // number of output features should match number of rows in params
        if (oShape[1] != wShape[1]) {
            return false;
        }
    }


    if (this->input != nullptr) {
        int *iShape = shape::shapeOf(this->inputShapeInfo);

        // we check for input/output batchSize equality
        if (oShape[0] != iShape[0])
            return false;
    }

    return true;
}

// feed forward
template<typename T, typename AF> void DenseLayer<T,AF>::feedForward() {
    // dropout helper call
    if (this->dropOut)
        this->dropOutHelper(this->input, this->inputShapeInfo);

    // dropconnect helper
    if (this->dropConnect)
        this->dropConnectHelper(this->params, this->paramsShapeInfo);
    

    // do wxa+b here or something else
    // TODO: introduce BLAS right here
    if (shape::isRowVector(this->inputShapeInfo)) {
        // gemv here input * W
    } else {
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
