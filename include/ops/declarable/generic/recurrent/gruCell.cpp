// 
// created by Yurii Shyrma on 05.12.2017
//

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/gruCell.h>

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

    NDArray<T>* Wx   = INPUT_VARIABLE(2);                   // input-to-hidden weights, [inSize   x 3*numUnits] 
    NDArray<T>* Wh   = INPUT_VARIABLE(3);                   // hidden-to-hidden weights, [numUnits x 3*numUnits]     
    NDArray<T>* b    = INPUT_VARIABLE(4);                   // biases, [3*numUnits] 
    
    NDArray<T>* ht   =  OUTPUT_VARIABLE(0);                  // current cell output [batchSize x numUnits], that is at current time step t    

    const int batchSize   = xt->sizeAt(0);
    const int inSize      = xt->sizeAt(1);
    const int numUnits    = ht_1->sizeAt(1);
    
    // input validation
    // check shapes of previous cell output 
    REQUIRE_TRUE(ht_1->sizeAt(0) == batchSize, 0, "CUSTOM_OP gruCell: the shape of previous cell output is wrong !");    
    // check shape of input-to-hidden weights
    REQUIRE_TRUE(Wx->isSameShape({inSize, 3*numUnits}), 0, "CUSTOM_OP gruCell: the shape of input-to-hidden weights is wrong !");
    // check shape of hidden-to-hidden weights
    REQUIRE_TRUE(Wh->isSameShape({numUnits, 3*numUnits}), 0, "CUSTOM_OP gruCell: the shape of hidden-to-hidden weights is wrong !");    
    // check shape of biases
    REQUIRE_TRUE(b->isSameShape({3*numUnits}), 0, "CUSTOM_OP gruCell: the shape of biases is wrong !");

    helpers::gruCell({xt, ht_1, Wx, Wh, b}, ht);

    return Status::OK();
}



DECLARE_SHAPE_FN(gruCell) {    
    
    // evaluate output shapeInfo
    int *outShapeInfo(nullptr);
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);
                
    outShapeInfo[0] = 2;
    outShapeInfo[1] = inputShape->at(0)[0];
    outShapeInfo[2] = inputShape->at(1)[1];
    
    shape::updateStrides(outShapeInfo, shape::order(inputShape->at(1)));
         
    return SHAPELIST(outShapeInfo);
}   








}
}

