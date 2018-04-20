//
//  Created by Yurii Shyrma on 20.01.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_svd)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/svd.h>

namespace nd4j {
namespace ops {

CUSTOM_OP_IMPL(svd, 1, 1, false, 0, 3) {
    
    NDArray<T>* x = INPUT_VARIABLE(0);
    NDArray<T>* s = OUTPUT_VARIABLE(0);
    
    const int rank =  x->rankOf();
    REQUIRE_TRUE(rank >= 2 , 0, "CUSTOM_OP svd: the rank of input array must be >=2 !");

    const bool fullUV = (bool)INT_ARG(0);
    const bool calcUV = (bool)INT_ARG(1);
    const int switchNum =  INT_ARG(2);    
    
    const int sRank = rank == 2 ? 2 : rank - 1; 

    ResultSet<T>* listX = NDArrayFactory<T>::allTensorsAlongDimension(x, {rank-2, rank-1});
    ResultSet<T>* listS = NDArrayFactory<T>::allTensorsAlongDimension(s, {sRank-1});
    ResultSet<T>* listU(nullptr), *listV(nullptr);
    
    if(calcUV) {
        NDArray<T>* u = OUTPUT_VARIABLE(1);
        NDArray<T>* v = OUTPUT_VARIABLE(2);
        listU = NDArrayFactory<T>::allTensorsAlongDimension(u, {rank-2, rank-1});
        listV = NDArrayFactory<T>::allTensorsAlongDimension(v, {rank-2, rank-1});
    }

    for(int i = 0; i < listX->size(); ++i) {
        
        // NDArray<T> matrix(x->ordering(), {listX->at(i)->sizeAt(0), listX->at(i)->sizeAt(1)}, block.getWorkspace());
        // matrix.assign(listX->at(i));
        helpers::SVD<T> svdObj(*(listX->at(i)), switchNum, calcUV, calcUV, fullUV);    
        listS->at(i)->assign(svdObj._s);

        if(calcUV) {
            listU->at(i)->assign(svdObj._u);
            listV->at(i)->assign(svdObj._v);
        }        
    }

    delete listX;
    delete listS;
    
    if(calcUV) {
        delete listU;
        delete listV;
    }
         
    return ND4J_STATUS_OK;
}


DECLARE_SHAPE_FN(svd) {

    int* inShapeInfo = inputShape->at(0);
    bool fullUV = (bool)INT_ARG(0);
    bool calcUV = (bool)INT_ARG(1);
    
    const int rank = inShapeInfo[0];
    const int diagSize = inShapeInfo[rank] < inShapeInfo[rank-1] ? inShapeInfo[rank] : inShapeInfo[rank-1];
        
    
    int* sShapeInfo(nullptr);
    if(rank == 2) {
        ALLOCATE(sShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int); 
        sShapeInfo[0] = 2;
        sShapeInfo[1] = 1;
        sShapeInfo[2] = diagSize;
    }
    else {
        ALLOCATE(sShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank-1), int); 
        sShapeInfo[0] = rank - 1;
        for(int i=1; i <= rank-2; ++i)
            sShapeInfo[i] = inShapeInfo[i];
        sShapeInfo[rank-1] = diagSize;
    }
    
    shape::updateStrides(sShapeInfo, shape::order(inShapeInfo));
    
    if(calcUV){

        int* uShapeInfo(nullptr), *vShapeInfo(nullptr);
        COPY_SHAPE(inShapeInfo, uShapeInfo);
        COPY_SHAPE(inShapeInfo, vShapeInfo);

        if(fullUV) {
            uShapeInfo[rank]   = uShapeInfo[rank-1];
            vShapeInfo[rank-1] = vShapeInfo[rank];
        }
        else {
            uShapeInfo[rank] = diagSize;
            vShapeInfo[rank-1] = vShapeInfo[rank];
            vShapeInfo[rank] = diagSize;
        }
    
        shape::updateStrides(uShapeInfo, shape::order(inShapeInfo));
        shape::updateStrides(vShapeInfo, shape::order(inShapeInfo));
    
        return SHAPELIST(sShapeInfo, uShapeInfo, vShapeInfo);        
    }         
    return SHAPELIST(sShapeInfo);
}


}
}

#endif