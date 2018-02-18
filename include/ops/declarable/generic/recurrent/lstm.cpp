//
// created by Yurii Shyrma on 15.02.2018
//

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/lstmCell.h>

namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstm, 8, 2, false, 3, 2) {

    NDArray<T>* x   = INPUT_VARIABLE(0);                   // input [time x batchSize x inSize]
    NDArray<T>* h0 = INPUT_VARIABLE(1);                    // initial cell output (at time step = 0) [batchSize x numProj], in case of projection=false -> numProj=numUnits!!! 
    NDArray<T>* c0 = INPUT_VARIABLE(2);                    // initial cell state  (at time step = 0) [batchSize x numUnits],  

    NDArray<T>* Wx  = INPUT_VARIABLE(3);                   // input-to-hidden  weights, [inSize  x 4*numUnits] 
    NDArray<T>* Wh  = INPUT_VARIABLE(4);                   // hidden-to-hidden weights, [numProj x 4*numUnits] 
    NDArray<T>* Wc  = INPUT_VARIABLE(5);                   // diagonal weights for peephole connections [3*numUnits] 
    NDArray<T>* Wp  = INPUT_VARIABLE(6);                   // projection weights [numUnits x numProj] 
    NDArray<T>* b   = INPUT_VARIABLE(7);                   // biases, [4*numUnits] 
    
    NDArray<T>* h   =  OUTPUT_VARIABLE(0);                 // cell outputs [time x batchSize x numProj], that is per each time step
    NDArray<T>* c   =  OUTPUT_VARIABLE(1);                 // cell states  [time x batchSize x numUnits] that is per each time step
    
    int peephole   = INT_ARG(0);                           // if 1, provide peephole connections
    int projection = INT_ARG(1);                           // if 1, then projection is performed, if false then numProj==numUnits is mandatory!!!!
    T clippingCellValue  = T_ARG(0);                       // clipping value for ct, if it is not equal to zero, then cell state is clipped
    T clippingProjValue  = T_ARG(1);                       // clipping value for projected ht, if it is not equal to zero, then projected cell output is clipped
    T forgetBias   = T_ARG(2);

    int time      = x->sizeAt(0);
    int batchSize = x->sizeAt(1);
    int inSize    = x->sizeAt(2);
    int numProj   = h0->sizeAt(1);
    int numUnits  = c0->sizeAt(1);    

     // input validation
    // check shapes of previous cell output and previous cell state
    for(int i = 1; i <=2; ++i)
        REQUIRE_TRUE((INPUT_VARIABLE(i))->sizeAt(0) == batchSize, 0, "CUSTOM_OP lstm: the shape[0] of initial cell output and initial cell state must be equal to batch size !");
    // check shape of input-to-hidden  weights            
    REQUIRE_TRUE((INPUT_VARIABLE(3))->isSameShape({inSize, 4*numUnits}),0,"CUSTOM_OP lstm: the shape of input-to-hidden weights is wrong !");
    // check shape of hidden-to-hidden  weights
    REQUIRE_TRUE((INPUT_VARIABLE(4))->isSameShape({numProj, 4*numUnits}),0,"CUSTOM_OP lstm: the shape of hidden-to-hidden weights is wrong !");
    // check shape of diagonal weights
    REQUIRE_TRUE((INPUT_VARIABLE(5))->isSameShape({3*numUnits}), 0, "CUSTOM_OP lstm: the shape of diagonal weights is wrong !");
    // check shape of projection weights
    REQUIRE_TRUE((INPUT_VARIABLE(6))->isSameShape({numUnits, numProj}), 0, "CUSTOM_OP lstm: the shape of projection weights is wrong !");
    // check shape of biases
    REQUIRE_TRUE((INPUT_VARIABLE(7))->isSameShape({4*numUnits}), 0, "CUSTOM_OP lstm: the shape of biases is wrong !");
    REQUIRE_TRUE(!(!projection && numUnits != numProj), 0, "CUSTOM_OP lstm: projection option is switched of, and in this case output dimensionality for the projection matrices (numProj) must be equal to number of units in lstmCell !");

    NDArray<T> currentH(h0);
    NDArray<T> currentC(c0);

    ResultSet<T>* xSubArrs = NDArrayFactory<T>::allExamples(x);
    ResultSet<T>* hSubArrs = NDArrayFactory<T>::allExamples(h);
    ResultSet<T>* cSubArrs = NDArrayFactory<T>::allExamples(c);

    // loop through time steps
    for (int t = 0; t < time; ++t) {

        helpers::lstmCell<T>({xSubArrs->at(t),&currentH,&currentC, Wx,Wh,Wc,Wp, b},   {hSubArrs->at(t),cSubArrs->at(t)},   {(T)peephole, (T)projection, clippingCellValue, clippingProjValue, forgetBias});
        currentH.assign(hSubArrs->at(t));
        currentC.assign(cSubArrs->at(t));
    }
    
    delete xSubArrs;
    delete hSubArrs;
    delete cSubArrs;

    return Status::OK();
}



DECLARE_SHAPE_FN(lstm) {    

    // evaluate output shapeInfos
    int *hShapeInfo(nullptr), *cShapeInfo(nullptr);
    ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);
    ALLOCATE(cShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);    
            
    hShapeInfo[0] = cShapeInfo[0] = 3;
    hShapeInfo[1] = cShapeInfo[1] = inputShape->at(0)[1];
    hShapeInfo[2] = cShapeInfo[2] = inputShape->at(0)[2];
    hShapeInfo[3] = inputShape->at(1)[2];
    cShapeInfo[3] = inputShape->at(2)[2];    
    
    shape::updateStrides(hShapeInfo, shape::order(inputShape->at(1)));    
    shape::updateStrides(cShapeInfo, shape::order(inputShape->at(2)));
         
    return new ShapeList({hShapeInfo, cShapeInfo});
}   








}
}

