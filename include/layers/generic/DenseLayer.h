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
        int feedForward();

        // back propagate
        int backPropagate();
       
        // This method should validate layer parameters & bias, and return TRUE if everything ok. FALSE otherwise      
        inline int validateParameters();

        // This method should validate input parameters, and return TRUE if everything ok. FALSE otherwise
        inline int validateInput();

        // This method should valudate output parameters, and return TRUE if everything is ok, FALSE otherwise        
        inline int validateOutput();

        // this method should validate memory/holders for BP pass
        inline int validateGradients();
};





/////// implementation part ///////    

// default constructor
template<typename T, typename AF> DenseLayer<T,AF>::DenseLayer() { 

}     

// back propagate
template<typename T, typename AF>
int DenseLayer<T,AF>::backPropagate() {
    // delta = dL/dz
    // epsilon = dL/da
    // delta = epsilon * da/dz = previous_params_T * previous_delta (*) da/dz

    // temporary save output pointers
    T *__buffer = this->output->_buffer;
    int *__shapeInfo = this->output->_shapeInfo;

    NDArray<T> *preOutput = new NDArray<T>(this->input->shapeOf()[0], this->params->shapeOf()[1], 'f');
    this->output->replacePointers(preOutput->_buffer, preOutput->_shapeInfo);
    this->feedForward();

    // put buffers back
    this->output->replacePointers(__buffer, __shapeInfo);

    NDArray<T> *delta = new NDArray<T>(preOutput);
    // calculate/fill delta

    ActivationsExecutioner<T>::template executeBP<AF>(preOutput, this->epsilon, delta);

    // gradient_on_param = delta * next_output
    auto iT = this->input->transpose();
    this->gemmHelper(iT, delta, this->gradientW, (T) 1.0f, (T) 0.0f);
    // gradient_on_bias = delta

    NDArray<T> *sumArr = delta->sum({0}); 
    this->gradientB->assign(sumArr);
    // calculate next epsilon


    // creating temp output array
    auto *oT = new NDArray<T>(this->params->shapeOf()[0], this->input->shapeOf()[0], 'f');
    delta->transposei();
    this->gemmHelper(this->params, delta, oT, (T) 1.0f, (T) 0.0f);
    //printf("O length: %i\n", this->output->lengthOf());
    oT->transposei();
    this->output->assign(oT);

    delete delta;
    delete sumArr;
    delete preOutput;
    delete oT;
    delete iT;

    return ND4J_STATUS_OK;
}



template<typename T, typename AF>
int DenseLayer<T,AF>::validateGradients() {
    if (this->gradientW == nullptr || this->gradientB == nullptr || this->bias == nullptr || !this->gradientW->nonNull() || !this->gradientB->nonNull() || !this->bias->nonNull())
        return ND4J_STATUS_BAD_GRADIENTS;
  // Gradient ret = new DefaultGradient();

    if (this->output == nullptr || !this->output->nonNull())
        return ND4J_STATUS_BAD_OUTPUT;
        // INDArray weightGrad = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY); //f order
        // Nd4j.gemm(input, delta, weightGrad, true, false, 1.0, 0.0);
        // INDArray biasGrad = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
        // delta.sum(biasGrad, 0); //biasGrad is initialized/zeroed first

        // ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
        // ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGrad);

    if (!this->gradientW->isSameShape(this->params)) {
        return ND4J_STATUS_BAD_GRADIENTS;
    }
        // INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(delta.transpose()).transpose();

    if (!this->gradientB->isSameShape(this->bias))
        return ND4J_STATUS_BAD_BIAS;
// # forward pass
// W = np.random.randn(5, 10)
// X = np.random.randn(10, 3)
// D = W.dot(X)

    // we're checking equality of input/epsilon batch size
    if (this->epsilon->shapeOf()[0] != this->input->shapeOf()[0])
        return ND4J_STATUS_BAD_EPSILON;
// # now suppose we had the gradient on D from above in the circuit
// dD = np.random.randn(*D.shape) # same shape as D
// dW = dD.dot(X.T) #.T gives the transpose of the matrix
// dX = W.T.dot(dD)



  // inline static T bpActivation(T value, T epsilon) {
                // // FIXME: ultra-bad. should consider conigurable extra params here
                // T extra[] = {(T) 0.0f};
                // return simdOps::Step<T>::template op(value, extra) * epsilon;

    if (this->epsilon->columns() != this->bias->columns())
        return ND4J_STATUS_BAD_EPSILON;

    // batch comparison again
    if (!this->output->isSameShape(this->input))
        return ND4J_STATUS_BAD_OUTPUT;

    return ND4J_STATUS_OK;
};



// This method should validate layer parameters & bias, and return TRUE if everything ok. FALSE otherwise
template<typename T, typename AF>
int DenseLayer<T,AF>::validateParameters() {
    if (this->params->_shapeInfo == nullptr || this->bias->_shapeInfo == nullptr || this->params == nullptr || this->bias == nullptr || this->params->_buffer == nullptr || this->bias->_buffer == nullptr) {
//        printf("Got nulls here\n");
        return ND4J_STATUS_BAD_PARAMS;
    }

    int wRank = this->params->rankOf();
    int bRank = this->bias->rankOf();

    // rank of params/bias has to be 2 here
    if (wRank != 2 || bRank != 2) {
//        printf("Non 2\n");
        return ND4J_STATUS_BAD_RANK;
    }


    int *wShape = this->params->shapeOf();

    int biasLength = this->bias->lengthOf();

    // number of outputs must be equal to biasLength
    if (wShape[1] != biasLength) {
//        printf("Bias doesn't match: %i vs %i\n", wShape[1], biasLength);
        return ND4J_STATUS_BAD_SHAPE;
    }


    return ND4J_STATUS_OK;
}


// This method should validate input parameters, and return TRUE if everything ok. FALSE otherwise
template<typename T, typename AF> int DenseLayer<T,AF>::validateInput() {
    // we expect input to be either vector or matrix, in both cases - that's rank2
    if (this->input == nullptr || this->input->_shapeInfo == nullptr ||this->input->_buffer == nullptr)
        return ND4J_STATUS_BAD_INPUT;

    if (this->input->rankOf() != 2)
        return ND4J_STATUS_BAD_RANK;


    int *iShape = this->input->shapeOf();

    if (this->params != nullptr && this->params->nonNull()) {
        // check dimensionality

        int *wShape = this->params->shapeOf();

        // number of input features should match number of rows in params
        if (iShape[1] != wShape[0]) {
            return ND4J_STATUS_BAD_SHAPE;
        }
    }

    if (this->output != nullptr && this->output->nonNull()) {
        int *oShape = this->output->shapeOf();

        // we check for input/output batchSize equality
        if (oShape[0] != iShape[0])
            return ND4J_STATUS_BAD_SHAPE;
    }

    return ND4J_STATUS_OK;
}


// This method should valudate output parameters, and return TRUE if everything is ok, FALSE otherwise
template<typename T, typename AF> int DenseLayer<T,AF>::validateOutput() {
    // same as input validation here. we expect rank of output arra
    if (this->output == nullptr || this->output->_buffer == nullptr || this->output->_shapeInfo == nullptr)
        return ND4J_STATUS_BAD_OUTPUT;

    if (this->output->rankOf() != 2)
        return ND4J_STATUS_BAD_RANK;

    int *oShape = this->output->shapeOf();

    // length of output along dimension 1 should match length of parameters, if parameters are set,
    if (this->params != nullptr && this->params->nonNull()) {
        int *wShape = this->params->shapeOf();

        // number of output features should match number of rows in params
        if (oShape[1] != wShape[1]) {
            return ND4J_STATUS_BAD_SHAPE;
        }
    }


    if (this->input != nullptr && this->input->nonNull()) {
        int *iShape = this->input->shapeOf();

        // we check for input/output batchSize equality
        if (oShape[0] != iShape[0])
            return ND4J_STATUS_BAD_SHAPE;
    }

    return ND4J_STATUS_OK;
}

// feed forward
template<typename T, typename AF> int DenseLayer<T,AF>::feedForward() {
    // dropout helper call
    if (this->dropOut) {
        //printf("Going dropout\n");
        this->dropOutHelper(this->input);
    }

    // dropconnect helper
    if (this->dropConnect) {
        //printf("Going dropconnect\n");
        this->dropConnectHelper(this->params);
    }
    

    // do wxa+b here or something else
    // TODO: introduce BLAS right here
    if (shape::isRowVector(this->input->_shapeInfo)) {
        // gemv here input * W

    } else {
        // gemm here, input * W
        // these values should be set appropriately

        this->gemmHelper(this->input, this->params, this->output, (T) 1.0f, (T) 0.0f);

        // we're rolling through rows here
        this->output->addiRowVector(this->bias);
    }

    // activation call
    ActivationsExecutioner<T>::template executeFF<AF>(this->output, this->output);

    return ND4J_STATUS_OK;
}


// end of namespace brackets
}
}

#endif //PROJECT_DENSE_H
