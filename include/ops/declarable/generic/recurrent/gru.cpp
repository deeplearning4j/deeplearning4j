//
// created by Yurii Shyrma on 15.02.2018
//

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/gruCell.h>

namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gru, 5, 1, false, 0, 0) {

    NDArray<T>* x  = INPUT_VARIABLE(0);                    // input [time x batchSize x inSize]
    NDArray<T>* h0 = INPUT_VARIABLE(1);                    // initial cell output (at time step = 0) [batchSize x numUnits], in case of projection=false -> numUnits=numUnits!!! 
    
    NDArray<T>* Wx  = INPUT_VARIABLE(2);                   // input-to-hidden  weights, [inSize x 3*numUnits] 
    NDArray<T>* Wh  = INPUT_VARIABLE(3);                   // hidden-to-hidden weights, [numUnits x 3*numUnits] 
    NDArray<T>* b   = INPUT_VARIABLE(4);                   // biases, [3*numUnits] 
    
    NDArray<T>* h   =  OUTPUT_VARIABLE(0);                 // cell outputs [time x batchSize x numUnits], that is per each time step            

    int time      = x->sizeAt(0);
    int batchSize = x->sizeAt(1);
    int inSize    = x->sizeAt(2);
    int numUnits  = h0->sizeAt(1);    

    // inputs validation
    // check shapes of initial cell output 
    REQUIRE_TRUE(h0->sizeAt(0) == batchSize, 0, "CUSTOM_OP gruCell: the shape of previous cell output is wrong !");    
    // check shape of input-to-hidden weights
    REQUIRE_TRUE(Wx->isSameShape({inSize, 3*numUnits}), 0, "CUSTOM_OP gruCell: the shape of input-to-hidden weights is wrong !");
    // check shape of hidden-to-hidden weights
    REQUIRE_TRUE(Wh->isSameShape({numUnits, 3*numUnits}), 0, "CUSTOM_OP gruCell: the shape of hidden-to-hidden weights is wrong !");    
    // check shape of biases
    REQUIRE_TRUE(b->isSameShape({3*numUnits}), 0, "CUSTOM_OP gruCell: the shape of biases is wrong !");

    NDArray<T> currentH(h0);

    ResultSet<T>* xSubArrs = NDArrayFactory<T>::allExamples(x);
    ResultSet<T>* hSubArrs = NDArrayFactory<T>::allExamples(h);
    
    // loop through time steps
    for (int t = 0; t < time; ++t) {

        helpers::gruCell<T>({xSubArrs->at(t),&currentH, Wx,Wh,b}, hSubArrs->at(t));
        currentH.assign(hSubArrs->at(t));    
    }
    
    delete xSubArrs;
    delete hSubArrs;    

    return Status::OK();
}



DECLARE_SHAPE_FN(gru) {    

    // evaluate output shapeInfos
    int *hShapeInfo(nullptr);
    ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);
            
    hShapeInfo[0] = 3;
    hShapeInfo[1] = inputShape->at(0)[1];
    hShapeInfo[2] = inputShape->at(0)[2];
    hShapeInfo[3] = inputShape->at(1)[2];
    
    shape::updateStrides(hShapeInfo, shape::order(inputShape->at(1)));        
         
    return SHAPELIST(hShapeInfo);
}   






}
}

