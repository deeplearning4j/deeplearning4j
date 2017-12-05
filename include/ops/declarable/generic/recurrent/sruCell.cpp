//
// implementation of operations for Simple Recurrent Unit: arXiv:1709.02755v2 [cs.CL] 12 Sep 2017
//
//  created by Yurii Shyrma on 05.12.2017
//

#include <ops/declarable/CustomOperations.h>


namespace nd4j {
    namespace ops {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static NDArray<T> activation(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Tanh<T>>();    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
static NDArray<T> sigmoid(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Sigmoid<T>>();    
}



//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sruCell, 4, 2, false, 0, 0) {

    NDArray<T>* xt   = INPUT_VARIABLE(0);               // input [batchSize x inSize], batchSize - batch size, inSize - number of features
    NDArray<T>* ct_1 = INPUT_VARIABLE(1);               // previous cell ct  [batchSize x inSize], that is at previous time step t-1   
    NDArray<T>* w    = INPUT_VARIABLE(2);               // weights [inSize x 3*inSize]
    NDArray<T>* b    = INPUT_VARIABLE(3);               // biases [1 × 2*inSize]

    NDArray<T>* ht   = OUTPUT_VARIABLE(0);              // current cell output [batchSize x inSize], that is at current time step t
    NDArray<T>* ct   = OUTPUT_VARIABLE(1);              // current cell state  [batchSize x inSize], that is at current time step t
    
    const int inSize    = xt->sizeAt(1);                // inSize - number of features
            
    NDArray<T> z = mmul(*xt, *w);                 //  [batchSize x 3*inSize]    

    // forget gate = sigmoid(xt*Wf + bf)
    NDArray<T> ft = sigmoid<T>(z({{},{inSize,   2*inSize}}) + (*b)({{},{0, inSize}}));
    
    // reset gate = sigmoid(xt*Wr + br)
    NDArray<T> rt = sigmoid<T>(z({{},{2*inSize, 3*inSize}}) + (*b)({{},{inSize, 2*inSize}}));

    // current sell state = ft(*)ct_1 + (1 - ft)(*)(*)(xt*Wc)
    *ct = ft*(*ct_1) + ((T)1. - ft)*z({{},{0, inSize}});
    // *ct = ft*(*ct_1 - z({},{0, inSize})) + z({{},{0, inSize}});

    // current cell output = rt(*)activation(ct) + (1 - rt)(*)xt
    *ht = rt*activation<T>(*ct) + ((T)1. - rt)*(*xt);
    // *ht = rt * (activation<T>(ct) - *xt) + *xt;
    
    return ND4J_STATUS_OK;
}


DECLARE_SHAPE_FN(sruCell) {

    const int batchSize = (INPUT_VARIABLE(0))->sizeAt(0);
    const int inSize    = (INPUT_VARIABLE(0))->sizeAt(1);

    // check shape of previous cell state    
    if((INPUT_VARIABLE(1))->sizeAt(0) != batchSize || (INPUT_VARIABLE(1))->sizeAt(1) != inSize)
        throw "CUSTOM_OP sruCell: the shape of previous cell state is wrong !";
    
    // check shape of weights
    if((INPUT_VARIABLE(2))->sizeAt(0) != inSize || (INPUT_VARIABLE(2))->sizeAt(1) != 3*inSize)
        throw "CUSTOM_OP sruCell: the shape of weights is wrong !";

    // check shape of biases
    if((INPUT_VARIABLE(3))->sizeAt(0) != 1 || (INPUT_VARIABLE(3))->sizeAt(1) != 2*inSize)
        throw "CUSTOM_OP sruCell: the shape of biases is wrong !";    

    // evaluate output shapeInfos
    int *outShapeInfo1(nullptr), *outShapeInfo2(nullptr);
    ALLOCATE(outShapeInfo1, block.getWorkspace(), 8, int);
    ALLOCATE(outShapeInfo2, block.getWorkspace(), 8, int);
            
    outShapeInfo1[0] = outShapeInfo2[0] = 2;
    outShapeInfo1[1] = outShapeInfo2[1] = batchSize;
    outShapeInfo1[2] = outShapeInfo2[2] = inSize;
    
    shape::updateStrides(outShapeInfo1, (INPUT_VARIABLE(0))->ordering());
    shape::updateStrides(outShapeInfo2, (INPUT_VARIABLE(0))->ordering());
         
    return new ShapeList({outShapeInfo1, outShapeInfo2});
}   




}
}
