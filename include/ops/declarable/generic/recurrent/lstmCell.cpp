//
// created by Yurii Shyrma on 30.11.2017
//

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/lstmCell.h>

namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstmCell, 8, 2, false, 3, 2) {

    NDArray<T>* xt   = INPUT_VARIABLE(0);                   // input [batchSize x inSize]
    NDArray<T>* ht_1 = INPUT_VARIABLE(1);                   // previous cell output [batchSize x numProj],  that is at previous time step t-1, in case of projection=false -> numProj=numUnits!!! 
    NDArray<T>* ct_1 = INPUT_VARIABLE(2);                   // previous cell state  [batchSize x numUnits], that is at previous time step t-1   

    NDArray<T>* Wx   = INPUT_VARIABLE(3);                   // input-to-hidden  weights, [inSize  x 4*numUnits] 
    NDArray<T>* Wh   = INPUT_VARIABLE(4);                   // hidden-to-hidden weights, [numProj x 4*numUnits] 
    NDArray<T>* Wc   = INPUT_VARIABLE(5);                   // diagonal weights for peephole connections [3*numUnits] 
    NDArray<T>* Wp   = INPUT_VARIABLE(6);                   // projection weights [numUnits x numProj] 
    NDArray<T>* b    = INPUT_VARIABLE(7);                   // biases, [4*numUnits] 
    
    NDArray<T>* ht   =  OUTPUT_VARIABLE(0);                 // current cell output [batchSize x numProj], that is at current time step t
    NDArray<T>* ct   =  OUTPUT_VARIABLE(1);                 // current cell state  [batchSize x numUnits], that is at current time step t
    
    int peephole   = INT_ARG(0);                            // if 1, provide peephole connections
    int projection = INT_ARG(1);                            // if 1, then projection is performed, if false then numProj==numUnits is mandatory!!!!
    T clippingCellValue  = T_ARG(0);                        // clipping value for ct, if it is not equal to zero, then cell state is clipped
    T clippingProjValue  = T_ARG(1);                        // clipping value for projected ht, if it is not equal to zero, then projected cell output is clipped
    T forgetBias   = T_ARG(2);

    int batchSize = xt->sizeAt(0);
    int inSize    = xt->sizeAt(1);
    int numProj   = ht_1->sizeAt(1);
    int numUnits  = ct_1->sizeAt(1);    

     // input validation
    // check shapes of previous cell output and previous cell state
    for(int i = 1; i <=2; ++i)
        REQUIRE_TRUE((INPUT_VARIABLE(i))->sizeAt(0) == batchSize, 0, "CUSTOM_OP lstmCell: the shape[0] of previous cell output and previous cell state must be equal to batch size !");
    // check shape of input-to-hidden  weights            
    REQUIRE_TRUE((INPUT_VARIABLE(3))->isSameShape({inSize, 4*numUnits}),0,"CUSTOM_OP lstmCell: the shape of input-to-hidden weights is wrong !");
    // check shape of hidden-to-hidden  weights
    REQUIRE_TRUE((INPUT_VARIABLE(4))->isSameShape({numProj, 4*numUnits}),0,"CUSTOM_OP lstmCell: the shape of hidden-to-hidden weights is wrong !");
    // check shape of diagonal weights
    REQUIRE_TRUE((INPUT_VARIABLE(5))->isSameShape({3*numUnits}), 0, "CUSTOM_OP lstmCell: the shape of diagonal weights is wrong !");
    // check shape of projection weights
    REQUIRE_TRUE((INPUT_VARIABLE(6))->isSameShape({numUnits, numProj}), 0, "CUSTOM_OP lstmCell: the shape of projection weights is wrong !");
    // check shape of biases
    REQUIRE_TRUE((INPUT_VARIABLE(7))->isSameShape({4*numUnits}), 0, "CUSTOM_OP lstmCell: the shape of biases is wrong !");
    REQUIRE_TRUE(!(!projection && numUnits != numProj), 0, "CUSTOM_OP lstmCell: projection option is switched of, and in this case output dimensionality for the projection matrices (numProj) must be equal to number of units in lstmCell !");

    // calculations
    helpers::lstmCell<T>({xt,ht_1,ct_1, Wx,Wh,Wc,Wp, b},   {ht,ct},   {(T)peephole, (T)projection, clippingCellValue, clippingProjValue, forgetBias});
    
    return Status::OK();
}



DECLARE_SHAPE_FN(lstmCell) {    

    // evaluate output shapeInfos
    int *outShapeInfo1(nullptr), *outShapeInfo2(nullptr);
    ALLOCATE(outShapeInfo1, block.getWorkspace(), 8, int);
    ALLOCATE(outShapeInfo2, block.getWorkspace(), 8, int);
            
    outShapeInfo1[0] = outShapeInfo2[0] = 2;
    outShapeInfo1[1] = outShapeInfo2[1] = (INPUT_VARIABLE(0))->sizeAt(0);
    outShapeInfo1[2] = (INPUT_VARIABLE(1))->sizeAt(1);
    outShapeInfo2[2] = (INPUT_VARIABLE(2))->sizeAt(1);    
    
    shape::updateStrides(outShapeInfo1, (INPUT_VARIABLE(1))->ordering());
    shape::updateStrides(outShapeInfo2, (INPUT_VARIABLE(2))->ordering());
         
    return SHAPELIST(outShapeInfo1, outShapeInfo2);
}   








}
}

