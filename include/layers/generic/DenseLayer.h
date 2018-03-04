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
  
        // feed forward
        virtual int feedForward();

        // back propagate
        virtual int backPropagate();
       
        // This method should validate layer parameters & bias, and return TRUE if everything ok. FALSE otherwise      
        inline virtual int validateParameters() const;

        // This method should validate input parameters, and return TRUE if everything ok. FALSE otherwise
        inline virtual int validateInput() const;

        // This method should valudate output parameters, and return TRUE if everything is ok, FALSE otherwise        
        inline virtual int validateOutput() const;

        // this method should validate memory/holders for BP pass
        inline virtual int validateGradients() const;
};


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///////////////////// implementation part ////////////////////////////

 
// back propagate
template<typename T, typename AF> int DenseLayer<T,AF>::backPropagate() {
    // delta = dL/dz
    // epsilon = dL/da
    // delta = epsilon * da/dz = previous_params_T * previous_delta (*) da/dz

    // temporary save output pointers
    T *buffer = this->_output->getBuffer();
    int *shapeInfo = this->_output->getShapeInfo();

    NDArray<T> *preOutput = new NDArray<T>('f',{this->_input->shapeOf()[0], this->_params->shapeOf()[1]});
    this->_output->replacePointers(preOutput->getBuffer(), preOutput->getShapeInfo());
    this->feedForward();

    // put buffers back
    this->_output->replacePointers(buffer, shapeInfo);

    NDArray<T> *delta = new NDArray<T>(preOutput);
    // calculate/fill delta

    ActivationsExecutioner<T>::template executeBP<AF>(preOutput, this->_epsilon, delta);

    // gradient_on_param = delta * next_output
    NDArray<T> *iT = this->_input->transpose();
    this->gemmHelper(iT, delta, this->_gradientW, (T) 1.0f, (T) 0.0f);
    // gradient_on_bias = delta

    // calculate biases gradients
    NDArray<T> *sumArr = delta->sum({0}); 
    this->_gradientB->assign(sumArr);
    // calculate next epsilon

    // creating temp output array
    NDArray<T> *oT = new NDArray<T>('f',{this->_params->shapeOf()[0], this->_input->shapeOf()[0]});
    delta->transposei();
    this->gemmHelper(this->_params, delta, oT, (T) 1.0f, (T) 0.0f);
    //printf("O length: %i\n", this->_output->lengthOf());
    oT->transposei();
    this->_output->assign(oT);

    delete preOutput;
    delete delta;
    delete sumArr;    
    delete oT;
    delete iT;

    return ND4J_STATUS_OK;
}



template<typename T, typename AF> int DenseLayer<T,AF>::validateGradients() const {
    
    if (this->_gradientW == nullptr || this->_gradientB == nullptr || this->_bias == nullptr || !this->_gradientW->nonNull() || !this->_gradientB->nonNull() || !this->_bias->nonNull())
        return ND4J_STATUS_BAD_GRADIENTS;    

    if (this->_output == nullptr || !this->_output->nonNull())
        return ND4J_STATUS_BAD_OUTPUT;
        
    if (!this->_gradientW->isSameShape(this->_params)) 
        return ND4J_STATUS_BAD_GRADIENTS;
    

    if (!this->_gradientB->isSameShape(this->_bias))
        return ND4J_STATUS_BAD_BIAS;

    // we're checking equality of input/epsilon batch size
    if (this->_epsilon->shapeOf()[0] != this->_input->shapeOf()[0])
        return ND4J_STATUS_BAD_EPSILON;

  // inline static T bpActivation(T value, T epsilon) {
                // // FIXME: ultra-bad. should consider conigurable extra params here
                // T extra[] = {(T) 0.0f};
                // return simdOps::Step<T>::template op(value, extra) * epsilon;

    if (this->_epsilon->columns() != this->_bias->columns())
        return ND4J_STATUS_BAD_EPSILON;

    // batch comparison again
    if (!this->_output->isSameShape(this->_input))
        return ND4J_STATUS_BAD_OUTPUT;

    return ND4J_STATUS_OK;
};



// This method should validate layer parameters & bias, and return TRUE if everything ok. FALSE otherwise
template<typename T, typename AF> int DenseLayer<T,AF>::validateParameters() const {
    if (this->_params->getShapeInfo() == nullptr || this->_bias->getShapeInfo() == nullptr || this->_params == nullptr || this->_bias == nullptr || this->_params->getBuffer() == nullptr || this->_bias->getBuffer() == nullptr) {
//        printf("Got nulls here\n");
        return ND4J_STATUS_BAD_PARAMS;
    }

    int wRank = this->_params->rankOf();
    int bRank = this->_bias->rankOf();

    // rank of params/bias has to be 2 here
    if (wRank != 2 || bRank != 2) {
//        printf("Non 2\n");
        return ND4J_STATUS_BAD_RANK;
    }


    int *wShape = this->_params->shapeOf();

    int biasLength = this->_bias->lengthOf();

    // number of outputs must be equal to biasLength
    if (wShape[1] != biasLength) {
//        printf("Bias doesn't match: %i vs %i\n", wShape[1], biasLength);
        return ND4J_STATUS_BAD_SHAPE;
    }


    return ND4J_STATUS_OK;
}


// This method should validate input parameters, and return TRUE if everything ok. FALSE otherwise
template<typename T, typename AF> int DenseLayer<T,AF>::validateInput() const {
    // we expect input to be either vector or matrix, in both cases - that's rank2
    if (this->_input == nullptr || this->_input->getShapeInfo() == nullptr ||this->_input->getBuffer() == nullptr)
        return ND4J_STATUS_BAD_INPUT;

    if (this->_input->rankOf() != 2)
        return ND4J_STATUS_BAD_RANK;


    int *iShape = this->_input->shapeOf();

    if (this->_params != nullptr && this->_params->nonNull()) {
        // check dimensionality

        int *wShape = this->_params->shapeOf();

        // number of input features should match number of rows in params
        if (iShape[1] != wShape[0]) {
            return ND4J_STATUS_BAD_SHAPE;
        }
    }

    if (this->_output != nullptr && this->_output->nonNull()) {
        int *oShape = this->_output->shapeOf();

        // we check for input/output batchSize equality
        if (oShape[0] != iShape[0])
            return ND4J_STATUS_BAD_SHAPE;
    }

    return ND4J_STATUS_OK;
}


// This method should valudate output parameters, and return TRUE if everything is ok, FALSE otherwise
template<typename T, typename AF> int DenseLayer<T,AF>::validateOutput() const {
    // same as input validation here. we expect rank of output arra
    if (this->_output == nullptr || this->_output->getBuffer() == nullptr || this->_output->getShapeInfo() == nullptr)
        return ND4J_STATUS_BAD_OUTPUT;

    if (this->_output->rankOf() != 2)
        return ND4J_STATUS_BAD_RANK;

    int *oShape = this->_output->shapeOf();

    // length of output along dimension 1 should match length of parameters, if parameters are set,
    if (this->_params != nullptr && this->_params->nonNull()) {
        int *wShape = this->_params->shapeOf();

        // number of output features should match number of rows in params
        if (oShape[1] != wShape[1]) {
            return ND4J_STATUS_BAD_SHAPE;
        }
    }


    if (this->_input != nullptr && this->_input->nonNull()) {
        int *iShape = this->_input->shapeOf();

        // we check for input/output batchSize equality
        if (oShape[0] != iShape[0])
            return ND4J_STATUS_BAD_SHAPE;
    }

    return ND4J_STATUS_OK;
}

// feed forward
template<typename T, typename AF> int DenseLayer<T,AF>::feedForward() {
    // dropout helper call
    if (this->_dropOut) {
        //printf("Going dropout\n");
        this->dropOutHelper(this->_input);
    }

    // dropconnect helper
    if (this->_dropConnect) {
        //printf("Going dropconnect\n");
        this->dropConnectHelper(this->_params);
    }
    

    // do wxa+b here or something else
    // TODO: introduce BLAS right here
    if (shape::isRowVector(this->_input->getShapeInfo())) {
        // gemv here input * W

    } else {
        // gemm here, input * W
        // these values should be set appropriately

        this->gemmHelper(this->_input, this->_params, this->_output, (T) 1.0f, (T) 0.0f);

        // we're rolling through rows here
        this->_output->addiRowVector(this->_bias);
    }

    // activation call
    ActivationsExecutioner<T>::template executeFF<AF>(this->_output, this->_output);

    return ND4J_STATUS_OK;
}


// end of namespace brackets
}
}

#endif //PROJECT_DENSE_H
