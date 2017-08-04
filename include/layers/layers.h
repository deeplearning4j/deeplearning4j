//
// @author raver119@gmail.com
//
#ifndef PROJECT_LAYERS_H
#define PROJECT_LAYERS_H

// the list of errors codes for layer data
#define ND4J_STATUS_OK 0
#define ND4J_STATUS_BAD_INPUT 1
#define ND4J_STATUS_BAD_SHAPE 2
#define ND4J_STATUS_BAD_RANK 3
#define ND4J_STATUS_BAD_PARAMS 4
#define ND4J_STATUS_BAD_OUTPUT 5


namespace nd4j {
namespace layers {

template <typename T> class INativeLayer {
    public:
        T   *params;                    // flattened rectangle matrix with parameters (weights) 
        int *paramsShapeInfo;           // defines matrix rank, numbers of elements per each dimension, dimensions strides, c-like or fortan-like order, element-wise-stride
                
        T *bias;                        // flattened multidimensional matrix of biases
        int *biasShapeInfo;               // see _paramsShapeInfo explanation
        
        T   *input;                     // flattened multidimensional matrix of inputs
        int *inputShapeInfo;            // see _paramsShapeInfo explanation

        T   *epsilon;                     // flattened multidimensional matrix of inputs
        int *epsilontShapeInfo;            // see _paramsShapeInfo explanation       
        
        T   *mask;                      // the matrix of zeros and unities, takes into account possible different size of inputs, for outer pixels absent in smaller inputs zeros are set, the rest is unities
        int *maskShapeInfo;             // see _paramsShapeInfo explanation

        T *output;                      // flattened multidimensional matrix of outputs
        int *outputShapeInfo;             // see _paramsShapeInfo explanation
        
        Nd4jIndex allocated;            // memory amount which is already used from workspace, more probably it would be just 0
        Nd4jIndex length;               // memory amount which is still available from workspace, (allocated + length) = total size of workspace
        void *workspace;                // if you are going to use additional memory, take it from workspace
        
        bool dropOut;                   // corresponds to dropout applying
        bool dropConnect;               // ???
        
        T pDropOut;                     // dropout probabilities (if used)
        T pDropConnect;                 // dropconnect probabilities (if used)
                
        int aNum;                       // activation function number (identity as default)

        // default constructor, sets all pointers to be empty
        INativeLayer();
        
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

        // This method should validate input parameters, and return corresponding codes errors if mistake is present
        virtual int validateInput() = 0;

        // This method should validate output parameters, and return TRUE if everything is ok, FALSE otherwise
        virtual int validateOutput() = 0;
       
        // DropOut & DropConnect helpers are platform-specific too
        virtual void dropOutHelper(T *input, int *shapeInfo) = 0;
        virtual void dropConnectHelper(T *input, int *shapeInfo) = 0;

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
            this->params = params;
            this->paramsShapeInfo = paramsShapeInfo;
            this->biasShapeInfo = biasShapeInfo;
            this->bias = bias;

            return validateParameters();
        }

        // We have some options to be configured in layer: dropout, dropconnect, lr, etc 
        // This method should handle that. Maybe map (key-value), or something like that?           
        int configureLayer(T *input, int *inputShapeInfo, T*output, int *outputShapeInfo, T pDropOut, T pDropConnect);

        // This inline method allows to specify input data for layer
        // this output will be either activation of this layer, or error from next layer        
        int setInput(T *input, int *inputShapeInfo, T *mask, int *maskShapeInfo) {
            this->input = input;
            this->inputShapeInfo = inputShapeInfo;
            this->mask = mask;
            this->maskShapeInfo = maskShapeInfo;

            return validateInput();
        }

        // This inline method allows to specify output pointer for layer
        int setOutput(T *output, int *shapeInfo) {
            this->output = output;
            this->outputShapeInfo = shapeInfo;

            return validateOutput();
        }

        // This method executes feed-forward pass on this layer
        virtual void feedForward() = 0;

        // This method executes back-propagation pass on this layer
        virtual void backPropagate() = 0;

        // gemv should be used here
        void gemvHelper(T *A, int *aShapeInfo, T *B, int *bShapeInfo, T *C, int *cShapeInfo, T alpha, T beta);

        void gemmHelper(NDArray<T> *A, NDArray<T> *B, NDArray<T> *C, T alpha, T beta);

        // extracts shapes info and perform gemm 
        void gemmHelper(T *A, int *aShapeInfo, T *B, int *bShapeInfo, T *C, int *cShapeInfo, T alpha, T beta);

};
    
/////// implementation part ///////
    
// default constructor sets all pointers to be empty
template <typename T> INativeLayer<T>::INativeLayer() {
    params = nullptr;   
    paramsShapeInfo = nullptr;
    bias = nullptr;    
    biasShapeInfo = nullptr;
    input = nullptr;
    inputShapeInfo = nullptr;
    epsilon = nullptr;
    epsilontShapeInfo; 
    mask = nullptr;
    maskShapeInfo = nullptr;
    output = nullptr;
    outputShapeInfo = nullptr;
    workspace = nullptr;
    Nd4jIndex allocated = 0;
    Nd4jIndex length = 0;
    dropOut = false;                   
    dropConnect = false;                       
    pDropOut = 0.;   
    pDropConnect = 0.;              
    aNum = 0;
}

template <typename T> void INativeLayer<T>::gemmHelper(NDArray<T> *A, NDArray<T> *B, NDArray<T> *C, T alpha, T beta) {
    gemmHelper(A->buffer, A->shapeInfo, B->buffer, B->shapeInfo, C->buffer, C->shapeInfo, alpha, beta);
}

// perform C = alpha*A*B + beta*C
template <typename T> void INativeLayer<T>::gemmHelper(T *A, int *aShapeInfo, T *B, int *bShapeInfo, T *C, int *cShapeInfo, T alpha, T beta) {  
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
    //auto *tC = new NDArray<T>(C, cShapeInfo);

    if (aOrder == bOrder) {
        rOrder = aOrder;

        if (aOrder == 'c') {
            // we might need to transpose matrices,     
            // todo: we need dup(c/f) helper here
        }

        if (rOrder == 'c') {
            M = cShape[1];
            N = cShape[0];
            K = aShape[1];
        } else {
            M = cShape[0];
            N = cShape[1];
            K = bShape[1];
        }

        lda = aShape[0];
        ldb = bShape[0];
        ldc = cShape[0];

        transA = 'N';
        transB = 'N';
    } else {
        // TODO: same dup('f) might be needed here, but obviously only one of operands
        printf("Going tRoute here\n");
        if (aOrder == 'c') {
            // dup(F) A here
            _A = tA->dup('f');
            _B = tB->dup('f');
        } else {
            // dup(F) B here
            _A = tA->dup('f');
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
    nd4j::blas::GEMM<T>::op(rOrder, false, false, M, N, K, alpha, _A->buffer, lda, _B->buffer, ldb, beta, C, ldc);

    delete tA;
    delete tB;
    //delete tC;

    delete _A;
    delete _B;
    //delete _C;
}



// We have some options to be configured in layer: dropout, dropconnect, lr, etc 
// This method should handle that. Maybe map (key-value), or something like that?           
template <typename T> int INativeLayer<T>::configureLayer(T *input, int *inputShapeInfo, T*output, int *outputShapeInfo, T pDropOut, T pDropConnect) {
    this->pDropOut = pDropOut > (T) 0.0f ? pDropOut : (T) 0.0f;
    this->pDropConnect = pDropConnect > (T) 0.0f ? pDropConnect : (T) 0.0f;

    this->dropOut = this->pDropOut > (T) 0.0f;
    this->dropConnect = this->pDropConnect > (T) 0.0f;

    this->input = input;
    this->inputShapeInfo = inputShapeInfo;

    if (validateInput() != ND4J_STATUS_OK)
        return ND4J_STATUS_BAD_INPUT;


    this->output = output;
    this->outputShapeInfo = outputShapeInfo;

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

