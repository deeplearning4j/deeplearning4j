//
// implementation of gated Recurrent Unit cell 
// (cf. http://arxiv.org/abs/1406.1078).
// Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
// "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
// 
// created by Yurii Shyrma on 05.12.2017
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static NDArray<T> _sigmoid(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Sigmoid<T>>();    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static NDArray<T> activation(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Tanh<T>>();    
}



//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gruCell, 5, 1, false, 0, 0) {

    NDArray<T>* xt   = INPUT_VARIABLE(0);                   // input [batchSize x inSize]
    NDArray<T>* ht_1 = INPUT_VARIABLE(1);                   // previous cell output [batchSize x numUnits],  that is at previous time step t-1

    NDArray<T>* Wx   = INPUT_VARIABLE(2);                   // input-to-hidden  weights, [inSize   x 3*numUnits] 
    NDArray<T>* Wh   = INPUT_VARIABLE(3);                   // hidden-to-hidden weights, [numUnits x 3*numUnits]     
    NDArray<T>* b    = INPUT_VARIABLE(4);                   // biases, [1 x 3*numUnits] 
    
    NDArray<T>* ht   =  OUTPUT_VARIABLE(0);                  // current cell output [batchSize x numUnits], that is at current time step t    

    const int batchSize   = (INPUT_VARIABLE(0))->sizeAt(0);
    const int inSize      = (INPUT_VARIABLE(0))->sizeAt(1);
    const int numUnits    = (INPUT_VARIABLE(1))->sizeAt(1);
    
    // input validation
    // check shapes of previous cell output 
    REQUIRE_TRUE((INPUT_VARIABLE(1))->sizeAt(0) == batchSize, 0, "CUSTOM_OP gruCell: the shape of previous cell output is wrong !");    
    // check shape of input-to-hidden weights
    REQUIRE_TRUE(!((INPUT_VARIABLE(2))->sizeAt(0) != inSize || (INPUT_VARIABLE(2))->sizeAt(1) != 3*numUnits), 0, "CUSTOM_OP gruCell: the shape of input-to-hidden weights is wrong !");
    // check shape of hidden-to-hidden weights
    REQUIRE_TRUE(!((INPUT_VARIABLE(3))->sizeAt(0) != numUnits || (INPUT_VARIABLE(3))->sizeAt(1) != 3*numUnits), 0, "CUSTOM_OP gruCell: the shape of hidden-to-hidden weights is wrong !");    
    // check shape of biases
    REQUIRE_TRUE(!((INPUT_VARIABLE(4))->sizeAt(0) != 1 || (INPUT_VARIABLE(4))->sizeAt(1) != 3*numUnits), 0, "CUSTOM_OP gruCell: the shape of biases is wrong !");


    // activ = sigmoid(xt*Wx + ht_1*Wh + b)
    NDArray<T> activ = _sigmoid<T>(mmul(*xt, (*Wx)({{},{0,2*numUnits}})) + mmul(*ht_1, (*Wh)({{},{0,2*numUnits}})) + (*b)({{},{0,2*numUnits}}));       // [batchSize x 2*numUnits] + [batchSize x 2*numUnits] + [1 x 2*numUnits] = [batchSize x 2*numUnits]    
    
    // reset gate
    NDArray<T> rt = activ({{}, {0, numUnits}});                     // [batchSize x numUnits]

    // update gate
    NDArray<T> ut = activ({{}, {numUnits, 2*numUnits}});            // [batchSize x numUnits]

    // ht_tilde = activation(xt*Wx + (rt(*)ht_1)*Wh + b)
    NDArray<T> ht_tilde = activation<T>(mmul(*xt, (*Wx)({{},{2*numUnits, 3*numUnits}})) + mmul((*ht_1)*rt, (*Wh)({{},{2*numUnits,3*numUnits}})) + (*b)({{},{2*numUnits,3*numUnits}}));     // [batchSize x numUnits]

    // current cell output
    *ht = ut * (*ht_1) + ((T)1. - ut) * ht_tilde;

    return ND4J_STATUS_OK;
}



DECLARE_SHAPE_FN(gruCell) {    
    
    // evaluate output shapeInfo
    int *outShapeInfo(nullptr);
    ALLOCATE(outShapeInfo, block.getWorkspace(), 8, int);
                
    outShapeInfo[0] = 2;
    outShapeInfo[1] = (INPUT_VARIABLE(0))->sizeAt(0);
    outShapeInfo[2] = (INPUT_VARIABLE(1))->sizeAt(1);    
    
    shape::updateStrides(outShapeInfo, (INPUT_VARIABLE(1))->ordering());
         
    return new ShapeList({outShapeInfo});
}   








}
}

