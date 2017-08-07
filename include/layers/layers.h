//
// @author raver119@gmail.com
//
#ifndef PROJECT_LAYERS_H
#define PROJECT_LAYERS_H

// the list of errors codes for layer data
#define ND4J_STATUS_OK         0
#define ND4J_STATUS_BAD_INPUT  1
#define ND4J_STATUS_BAD_SHAPE  2
#define ND4J_STATUS_BAD_RANK   3
#define ND4J_STATUS_BAD_PARAMS 4
#define ND4J_STATUS_BAD_OUTPUT 5
#define ND4J_STATUS_BAD_RNG 6
#define ND4J_STATUS_BAD_EPSILON 7
#define ND4J_STATUS_BAD_GRADIENTS 8
#define ND4J_STATUS_BAD_BIAS 9


namespace nd4j {
namespace layers {

template <typename T> class INativeLayer {
    public:

        NDArray<T> *params;
        //T   *params;                    // flattened rectangle matrix with parameters (weights)
        //int *paramsShapeInfo;           // defines matrix rank, numbers of elements per each dimension, dimensions strides, c-like or fortan-like order, element-wise-stride

        NDArray<T> *bias;
        //T *bias;                        // flattened multidimensional matrix of biases
        //int *biasShapeInfo;               // see _paramsShapeInfo explanation


        NDArray<T> *input;
        //T   *input;                     // flattened multidimensional matrix of inputs
        //int *inputShapeInfo;            // see _paramsShapeInfo explanation

        NDArray<T> *epsilon;
        //T   *epsilon;                     // epsilon = dL/da, L - loss function, a - activation
        //int *epsilontShapeInfo;            // see _paramsShapeInfo explanation

        NDArray<T> *mask;
        //T   *mask;                      // the matrix of zeros and unities, takes into account possible different size of inputs, for outer pixels absent in smaller inputs zeros are set, the rest is unities
        //int *maskShapeInfo;             // see _paramsShapeInfo explanation

        NDArray<T> *output;
        //T *output;                      // flattened multidimensional matrix of outputs
        //int *outputShapeInfo;             // see _paramsShapeInfo explanation

        NDArray<T> *gradientW;              // flattened multidimensional matrix of gradients used in bp
        NDArray<T> *gradientB;              // bias gradients holder
        
        Nd4jIndex allocated;            // memory amount which is already used from workspace, more probably it would be just 0
        Nd4jIndex length;               // memory amount which is still available from workspace, (allocated + length) = total size of workspace
        void *workspace;                // if you are going to use additional memory, take it from workspace

        nd4j::random::RandomBuffer *rng;    // rng helper

        bool dropOut;                   // corresponds to dropout applying
        bool dropConnect;               // ???
        
        T pDropOut;                     // dropout probabilities (if used)
        T pDropConnect;                 // dropconnect probabilities (if used)
                
        // default constructor, sets all pointers to be empty
        INativeLayer();

        ~INativeLayer();

        // copy constructor
        // creation of this class objects by copying is not expected, therefore disable copy constructor 
        INativeLayer(const INativeLayer& ) = delete;
        
        // assignment operator
        // the assignment operations are not expected for this class objects, therefore disable assignment operator
        INativeLayer& operator=(const INativeLayer& ) = delete;   

        // This method "allocates" memory chunk from workspace
        virtual T* allocate(long bytes) = 0; 
        
        // This method should validate parameters & bias, and return TRUE if everything ok. False otherwise
        virtual int validateParameters() = 0;

        // this method should validate memory/holders for BP pass
        virtual int validateGradients() = 0;

        // This method should validate input parameters, and return corresponding codes errors if mistake is present
        virtual int validateInput() = 0;

        // This method should validate output parameters, and return TRUE if everything is ok, FALSE otherwise
        virtual int validateOutput() = 0;
       
        // DropOut & DropConnect helpers are platform-specific too
        virtual void dropOutHelper(NDArray<T> *input) = 0;
        virtual void dropConnectHelper(NDArray<T> *input) = 0;

        // this inline method attaches layer to workspace memory
        void setWorkspace(void *memory, Nd4jIndex length) {
            this->length    = length;
            this->workspace = memory;
        };

        // this method returns number of bytes used
        long inline getUsedMemory() {
            return allocated;           // usually just 0
        }

        // This inline method allows to set parameters/biases for current layer
        // this input will be either activation from previous layer, or error coming from next layer
        int setParameters(T *params, int *paramsShapeInfo, T *bias, int *biasShapeInfo) {
            this->params->replacePointers(params, paramsShapeInfo);
            this->bias->replacePointers(bias, biasShapeInfo);

            return validateParameters();
        }

        // We have some options to be configured in layer: dropout, dropconnect, lr, etc 
        // This method should handle that. Maybe map (key-value), or something like that?           
        int configureLayerFF(T *input, int *inputShapeInfo, T*output, int *outputShapeInfo, T pDropOut, T pDropConnect, Nd4jPointer rngPointer);

        int configureLayerBP(T *output, int *outputShapeInfo, T* gradientW, int *gradientWShapeInfo, T* gradientB, int *gradientBShapeInfo, T *epsilonPrev, int *epsilonShapeInfo);

        // This inline method allows to specify input data for layer
        // this output will be either activation of this layer, or error from next layer        
        int setInput(T *input, int *inputShapeInfo, T *mask, int *maskShapeInfo) {
            this->input->replacePointers(input, inputShapeInfo);
            this->mask->replacePointers(mask, maskShapeInfo);

            return validateInput();
        }

        // This inline method allows to specify output pointer for layer
        int setOutput(T *output, int *shapeInfo) {
            this->output->replacePointers(output, shapeInfo);

            return validateOutput();
        }

        // This method executes feed-forward pass on this layer
        virtual int feedForward() = 0;

        // This method executes back-propagation pass on this layer
        virtual int backPropagate() = 0;

        // gemv should be used here
        void gemvHelper(T *A, int *aShapeInfo, T *B, int *bShapeInfo, T *C, int *cShapeInfo, T alpha, T beta);

        void gemmHelper(NDArray<T> *A, NDArray<T> *B, NDArray<T> *C, T alpha, T beta);

        // extracts shapes info and perform gemm 
        void gemmHelper(T *A, int *aShapeInfo, T *B, int *bShapeInfo, T *C, int *cShapeInfo, T alpha, T beta);

};
    
/////// implementation part ///////
    
// default constructor sets all pointers to be empty
template <typename T>
INativeLayer<T>::INativeLayer() {
    params = new NDArray<T>(nullptr, nullptr);
    bias = new NDArray<T>(nullptr, nullptr);
    input = new NDArray<T>(nullptr, nullptr);
    epsilon = new NDArray<T>(nullptr, nullptr);
    mask = new NDArray<T>(nullptr, nullptr);
    output = new NDArray<T>(nullptr, nullptr);
    epsilon = new NDArray<T>(nullptr, nullptr);
    gradientW = new NDArray<T>(nullptr, nullptr);
    gradientB = new NDArray<T>(nullptr, nullptr);

    workspace = nullptr;
    Nd4jIndex allocated = 0;
    Nd4jIndex length = 0;
    dropOut = false;                   
    dropConnect = false;                       
    pDropOut = 0.;   
    pDropConnect = 0.;
    rng = nullptr;
}

template <typename T>
INativeLayer<T>::~INativeLayer() {
    delete params;
    delete bias;
    delete input;
    delete gradientW;
    delete gradientB;
    delete mask;
    delete output;
    delete epsilon;
}

template <typename T> void INativeLayer<T>::gemmHelper(NDArray<T> *A, NDArray<T> *B, NDArray<T> *C, T alpha, T beta) {
    gemmHelper(A->buffer, A->shapeInfo, B->buffer, B->shapeInfo, C->buffer, C->shapeInfo, alpha, beta);
}

// perform C = alpha*A*B + beta*C
template <typename T> void INativeLayer<T>::gemmHelper(T *A, int *aShapeInfo, T *B, int *bShapeInfo, T *C, int *cShapeInfo, T alpha, T beta) {
            /**
             * PLEASE NOTE: Return order will be F always
             */
    char aOrder = shape::order(aShapeInfo);
    char bOrder = shape::order(bShapeInfo);
    char cOrder = shape::order(cShapeInfo);

    int *aShape = shape::shapeOf(aShapeInfo);
    int *bShape = shape::shapeOf(bShapeInfo);
    int *cShape = shape::shapeOf(cShapeInfo);

    char rOrder;

    int M, N, K, lda, ldb, ldc;
    char transA, transB;

    NDArray<T> *_A, *_B, *_C;

    //_C = new NDArray<T>(C, cShapeInfo);

    auto *tA = new NDArray<T>(A, aShapeInfo);
    auto *tB = new NDArray<T>(B, bShapeInfo);
    auto *tC = new NDArray<T>(C, cShapeInfo);

    if (cOrder != 'f') {
        _C = tC->dup('f');
    } else {
        _C = tC;
    }

    if (aOrder == bOrder) {
        //printf("Going dRoute here\n");

        if (aOrder == 'c') {
            // we might need to transpose matrices,     
            // todo: we need dup(c/f) helper here
            _A = tA->dup('f');
            _B = tB->dup('f');
        } else {
            _A = tA;
            _B = tB;
        }

        rOrder = 'f';

        M = cShape[0];
        N = cShape[1];
        K = aShape[1];

        lda = aShape[0];
        ldb = bShape[0];
        ldc = cShape[0];

        transA = 'N';
        transB = 'N';
    } else {
        //printf("Going tRoute here\n");
        if (aOrder == 'c') {
            // dup(F) A here
            _A = tA->dup('f');
            _B = tB;
        } else {
            // dup(F) B here
            _A = tA;
            _B = tB->dup('f');
        }

       // _C = tC->dup('f');

        M = cShape[0];
        N = cShape[1];
        K = aShape[1]; 

        rOrder = aOrder;

        lda = aShape[0];
        ldb = bShape[0];
        ldc = cShape[0];

        transA = 'N';
        transB = 'N';
    }

    // we'll use platform-specific gemm here eventually. maybe tomorrow.
    // TODO: put proper _gemm here
    nd4j::blas::GEMM<T>::op(rOrder, transA, transB, M, N, K, alpha, _A->buffer, lda, _B->buffer, ldb, beta, _C->buffer, ldc);

    if (cOrder != 'f') {
        tC->assign(_C);
    }

    if (tA != _A)
        delete _A;

    if (tB != _B)
        delete _B;

    if (tC != _C)
        delete _C;


    delete tA;
    delete tB;
    delete tC;
}


template <typename T>
int INativeLayer<T>::configureLayerBP(T *output, int *outputShapeInfo, T* gradientW, int *gradientWShapeInfo, T* gradientB, int *gradientBShapeInfo, T *epsilonPrev, int *epsilonShapeInfo) {
    this->output->replacePointers(output, outputShapeInfo);
    this->gradientW->replacePointers(gradientW, gradientWShapeInfo);
    this->gradientB->replacePointers(gradientB, gradientBShapeInfo);
    this->epsilon->replacePointers(epsilonPrev, epsilonShapeInfo);

    // TODO: add gradient/epsilon valdiation here
    if (validateGradients() != ND4J_STATUS_OK)
        return validateGradients();

    return ND4J_STATUS_OK;
}



// We have some options to be configured in layer: dropout, dropconnect, lr, etc 
// This method should handle that. Maybe map (key-value), or something like that?           
template <typename T>
int INativeLayer<T>::configureLayerFF(T *input, int *inputShapeInfo, T*output, int *outputShapeInfo, T pDropOut, T pDropConnect, Nd4jPointer ptrRng) {

    if (ptrRng != nullptr)
        this->rng = reinterpret_cast<nd4j::random::RandomBuffer *> (ptrRng);

    this->pDropOut = pDropOut > (T) 0.0f ? pDropOut : (T) 0.0f;
    this->pDropConnect = pDropConnect > (T) 0.0f ? pDropConnect : (T) 0.0f;

    this->dropOut = this->pDropOut > (T) 0.0f;
    this->dropConnect = this->pDropConnect > (T) 0.0f;

    if ((this->dropOut || this->dropConnect) && this->rng == nullptr)
        return ND4J_STATUS_BAD_RNG;

    this->input->replacePointers(input, inputShapeInfo);


    if (validateInput() != ND4J_STATUS_OK)
        return ND4J_STATUS_BAD_INPUT;


    this->output->replacePointers(output, outputShapeInfo);

    if (validateOutput() != ND4J_STATUS_OK)
        return ND4J_STATUS_BAD_OUTPUT;

    /*
     * TODO: define ERROR_CODES here, and return them instead of bool
     */

    return ND4J_STATUS_OK;
}          


// end of namespace brackets
}
}    
#endif //PROJECT_LAYERS_H

